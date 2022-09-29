#include <iostream>
#define LANTERN_BUILD
#include <c10/core/CPUAllocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/torch.h>

#include <thread>

#include "AllocatorUtils.h"
#include "lantern/lantern.h"
#include "utils.hpp"

enum CallGC {
  lite,
  full,
  no,
};

CallGC should_call_gc () {
  
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  
  size_t device_free;
  size_t device_total;
  C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
  
  auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
  
  auto current_reserved = stats.reserved_bytes[0].current; 
  auto current_allocated = stats.allocated_bytes[0].current;
  
  auto reserved_rate = (double)current_reserved/(double)device_total;
  if (reserved_rate < cuda_allocator_reserved_rate) {
    return CallGC::no;
  }
  
  auto allocated_rate = (double)current_allocated/(double)device_total;
  if (allocated_rate > cuda_allocator_allocated_rate) {
    return CallGC::full;
  }
  
  auto allocated_over_reserved_rate = (double)current_allocated/(double)current_reserved;
  if (allocated_over_reserved_rate > cuda_allocator_allocated_reserved_rate) {
    return CallGC::full;
  }
  
  return CallGC::lite;
}

namespace c10 {
class GarbageCollectorCallback : virtual public c10::FreeMemoryCallback {
 public:
  bool Execute() {
    switch (should_call_gc()) {
    case CallGC::no:
      return false;
    case CallGC::lite:
      (*call_r_gc)(false);  
      break;
    case CallGC::full:
      (*call_r_gc)(true);  
      break;
    }
    wait_for_gc();
    return true;
  }
};

REGISTER_FREE_MEMORY_CALLBACK("garbage_collector_callback",
                              GarbageCollectorCallback)
}  // namespace c10