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
    ALLOCATION_FAILED // buffer allocation failed
  };
  virtual ~IMpsAllocatorCallback() = default;
  virtual void executeMPSAllocatorCallback(void* ptr, EventType event) = 0;
};

// MPS allocator will execute every registered callback when a block of memory is freed.
C10_DECLARE_REGISTRY(MPSAllocatorCallbacksRegistry, IMpsAllocatorCallback);
#define REGISTER_MPS_ALLOCATOR_CALLBACK(name, ...) \
C10_REGISTER_CLASS(MPSAllocatorCallbacksRegistry, name, __VA_ARGS__);

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
        // this is never used currently.
            break;
        case EventType::ALLOCATION_FAILED:
        // we want to call the gc in this situation:
        // https://github.com/pytorch/pytorch/blob/b37a50afda55c5b73298016d10fca1f8c6f65055/aten/src/ATen/mps/MPSAllocator.mm#L211C44-L211C78
            (*call_r_gc)(true);
            wait_for_gc();
            break;
        default:
            break;
     }
    }
};

REGISTER_MPS_ALLOCATOR_CALLBACK("gc", MPSGarbageCollectorCallback);

}}

