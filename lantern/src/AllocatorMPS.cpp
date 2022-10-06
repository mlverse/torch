#include <iostream>
#define LANTERN_BUILD
#include <torch/torch.h>
#include "AllocatorUtils.h"

// this implementation is based on CUDACachingAllocator.
// It utilizes Metal Heaps to improve the performance with buffer allocation.
// TODO: Unify the logic with CUDACachingAllocator and remove redundant code.
namespace at {
namespace mps {


class IMpsAllocatorCallback {
 public:
  enum class EventType {
    ALLOCATED, // buffer got allocated to be used immediately
    RECYCLED,  // buffer pulled from free list to be reused
    FREED,     // buffer put to free list for future recycling
    RELEASED,  // buffer memory released
  };
  virtual ~IMpsAllocatorCallback() = default;
  virtual void executeMPSAllocatorCallback(void* ptr, EventType event) = 0;
};

// MPS allocator will execute every registered callback when a block of memory is freed.
C10_DECLARE_REGISTRY(MPSAllocatorCallbacksRegistry, IMpsAllocatorCallback);
#define REGISTER_MPS_ALLOCATOR_CALLBACK(name, ...) \
  C10_REGISTER_CLASS(MPSAllocatorCallbacksRegistry, name, __VA_ARGS__);

at::Allocator* getMPSStaticAllocator();

int free_calls = 0;

class MPSGarbageCollectorCallback : virtual public at::mps::IMpsAllocatorCallback {
    public:
    MPSGarbageCollectorCallback() = default;
    ~MPSGarbageCollectorCallback() = default;
    void executeMPSAllocatorCallback(void* ptr, EventType event) {
     switch (event)
     {
        case EventType::ALLOCATED:
        // seems to be currently unreachable
            break;
        case EventType::RECYCLED:
        // seems to be currently unreachable
            break;
        case EventType::FREED:
        // caling gc here will deadlock.
            break;
        case EventType::RELEASED:
        // we want to call the gc in this situation:
        // https://github.com/pytorch/pytorch/blob/664058fa83f1d8eede5d66418abff6e20bd76ca8/aten/src/ATen/mps/MPSAllocator.mm#L215
            (*call_r_gc)(true);
            wait_for_gc();
            break;
        default:
            break;
     }
    }
};

REGISTER_MPS_ALLOCATOR_CALLBACK("gc", MPSGarbageCollectorCallback);

}
}

