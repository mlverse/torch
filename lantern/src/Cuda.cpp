#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"
#ifdef __NVCC__
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/CUDAHooks.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#include <torch/torch.h>

#include "utils.hpp"

bool _lantern_cuda_is_available() {
  LANTERN_FUNCTION_START
  return torch::cuda::is_available();
  LANTERN_FUNCTION_END_RET(false)
}

int _lantern_cuda_device_count() {
  LANTERN_FUNCTION_START
  return torch::cuda::device_count();
  LANTERN_FUNCTION_END_RET(0)
}

int64_t _lantern_cuda_current_device() {
  LANTERN_FUNCTION_START
  return at::detail::getCUDAHooks().current_device();
  LANTERN_FUNCTION_END_RET(0)
}

void _lantern_cuda_show_config() {
  LANTERN_FUNCTION_START
  std::cout << at::detail::getCUDAHooks().showConfig() << std::endl;
  LANTERN_FUNCTION_END_VOID
}

void* _lantern_cuda_get_device_capability(int64_t device) {
  LANTERN_FUNCTION_START
#ifdef __NVCC__
  cudaDeviceProp* devprop = at::cuda::getDeviceProperties(device);
  std::vector<int64_t> cap = {devprop->major, devprop->minor};
  return make_raw::vector::int64_t(cap);
#else
  throw std::runtime_error(
      "`cuda_get_device` is only supported on CUDA runtimes.");
#endif
  LANTERN_FUNCTION_END
}

int64_t _lantern_cudnn_runtime_version() {
  return at::detail::getCUDAHooks().versionCuDNN();
}

bool _lantern_cudnn_is_available() {
  return at::detail::getCUDAHooks().hasCuDNN();
}

void* _lantern_cuda_device_stats(int64_t device) {
  LANTERN_FUNCTION_START
#ifdef __NVCC__
  auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
  std::vector<int64_t> results;
  results.push_back(stats.num_alloc_retries);
  results.push_back(stats.num_ooms);
  results.push_back(stats.max_split_size);

  results.push_back(stats.oversize_allocations.current);
  results.push_back(stats.oversize_allocations.peak);
  results.push_back(stats.oversize_allocations.allocated);
  results.push_back(stats.oversize_allocations.freed);

  results.push_back(stats.oversize_segments.current);
  results.push_back(stats.oversize_segments.peak);
  results.push_back(stats.oversize_segments.allocated);
  results.push_back(stats.oversize_segments.freed);

  results.push_back(stats.allocation[0].current);
  results.push_back(stats.allocation[0].peak);
  results.push_back(stats.allocation[0].allocated);
  results.push_back(stats.allocation[0].freed);
  results.push_back(stats.allocation[1].current);
  results.push_back(stats.allocation[1].peak);
  results.push_back(stats.allocation[1].allocated);
  results.push_back(stats.allocation[1].freed);
  results.push_back(stats.allocation[2].current);
  results.push_back(stats.allocation[2].peak);
  results.push_back(stats.allocation[2].allocated);
  results.push_back(stats.allocation[2].freed);

  results.push_back(stats.segment[0].current);
  results.push_back(stats.segment[0].peak);
  results.push_back(stats.segment[0].allocated);
  results.push_back(stats.segment[0].freed);
  results.push_back(stats.segment[1].current);
  results.push_back(stats.segment[1].peak);
  results.push_back(stats.segment[1].allocated);
  results.push_back(stats.segment[1].freed);
  results.push_back(stats.segment[2].current);
  results.push_back(stats.segment[2].peak);
  results.push_back(stats.segment[2].allocated);
  results.push_back(stats.segment[2].freed);

  results.push_back(stats.active[0].current);
  results.push_back(stats.active[0].peak);
  results.push_back(stats.active[0].allocated);
  results.push_back(stats.active[0].freed);
  results.push_back(stats.active[1].current);
  results.push_back(stats.active[1].peak);
  results.push_back(stats.active[1].allocated);
  results.push_back(stats.active[1].freed);
  results.push_back(stats.active[2].current);
  results.push_back(stats.active[2].peak);
  results.push_back(stats.active[2].allocated);
  results.push_back(stats.active[2].freed);

  results.push_back(stats.inactive_split[0].current);
  results.push_back(stats.inactive_split[0].peak);
  results.push_back(stats.inactive_split[0].allocated);
  results.push_back(stats.inactive_split[0].freed);
  results.push_back(stats.inactive_split[1].current);
  results.push_back(stats.inactive_split[1].peak);
  results.push_back(stats.inactive_split[1].allocated);
  results.push_back(stats.inactive_split[1].freed);
  results.push_back(stats.inactive_split[2].current);
  results.push_back(stats.inactive_split[2].peak);
  results.push_back(stats.inactive_split[2].allocated);
  results.push_back(stats.inactive_split[2].freed);

  results.push_back(stats.allocated_bytes[0].current);
  results.push_back(stats.allocated_bytes[0].peak);
  results.push_back(stats.allocated_bytes[0].allocated);
  results.push_back(stats.allocated_bytes[0].freed);
  results.push_back(stats.allocated_bytes[1].current);
  results.push_back(stats.allocated_bytes[1].peak);
  results.push_back(stats.allocated_bytes[1].allocated);
  results.push_back(stats.allocated_bytes[1].freed);
  results.push_back(stats.allocated_bytes[2].current);
  results.push_back(stats.allocated_bytes[2].peak);
  results.push_back(stats.allocated_bytes[2].allocated);
  results.push_back(stats.allocated_bytes[2].freed);

  results.push_back(stats.reserved_bytes[0].current);
  results.push_back(stats.reserved_bytes[0].peak);
  results.push_back(stats.reserved_bytes[0].allocated);
  results.push_back(stats.reserved_bytes[0].freed);
  results.push_back(stats.reserved_bytes[1].current);
  results.push_back(stats.reserved_bytes[1].peak);
  results.push_back(stats.reserved_bytes[1].allocated);
  results.push_back(stats.reserved_bytes[1].freed);
  results.push_back(stats.reserved_bytes[2].current);
  results.push_back(stats.reserved_bytes[2].peak);
  results.push_back(stats.reserved_bytes[2].allocated);
  results.push_back(stats.reserved_bytes[2].freed);

  results.push_back(stats.active_bytes[0].current);
  results.push_back(stats.active_bytes[0].peak);
  results.push_back(stats.active_bytes[0].allocated);
  results.push_back(stats.active_bytes[0].freed);
  results.push_back(stats.active_bytes[1].current);
  results.push_back(stats.active_bytes[1].peak);
  results.push_back(stats.active_bytes[1].allocated);
  results.push_back(stats.active_bytes[1].freed);
  results.push_back(stats.active_bytes[2].current);
  results.push_back(stats.active_bytes[2].peak);
  results.push_back(stats.active_bytes[2].allocated);
  results.push_back(stats.active_bytes[2].freed);

  results.push_back(stats.inactive_split_bytes[0].current);
  results.push_back(stats.inactive_split_bytes[0].peak);
  results.push_back(stats.inactive_split_bytes[0].allocated);
  results.push_back(stats.inactive_split_bytes[0].freed);
  results.push_back(stats.inactive_split_bytes[1].current);
  results.push_back(stats.inactive_split_bytes[1].peak);
  results.push_back(stats.inactive_split_bytes[1].allocated);
  results.push_back(stats.inactive_split_bytes[1].freed);
  results.push_back(stats.inactive_split_bytes[2].current);
  results.push_back(stats.inactive_split_bytes[2].peak);
  results.push_back(stats.inactive_split_bytes[2].allocated);
  results.push_back(stats.inactive_split_bytes[2].freed);

  return make_raw::vector::int64_t(results);
#else
  throw std::runtime_error(
      "`cuda_device_stats` is only supported on CUDA runtimes.");
#endif
  LANTERN_FUNCTION_END
}

int _lantern_cuda_get_runtime_version() {
  LANTERN_FUNCTION_START
#ifdef __NVCC__
  int runtimeVersion;
  cudaRuntimeGetVersion(&runtimeVersion);
  return runtimeVersion;
#else
  throw std::runtime_error(
      "`cuda_device_stats` is only supported on CUDA runtimes.");
#endif
  LANTERN_FUNCTION_END
}