#include <iostream>
#define LANTERN_BUILD
#include "lantern/lantern.h"
#include <torch/torch.h>
#include <thread>
#include <c10/core/CPUAllocator.h>
#include "utils.hpp"
#include "AllocatorUtils.h"

#include <c10/cuda/CUDACachingAllocator.h>

namespace c10
{
class GarbageCollectorCallback : virtual public c10::FreeMemoryCallback {
public:

  bool Execute() {

    if (std::this_thread::get_id() == MAIN_THREAD_ID)
    {
      (*call_r_gc)();
    }

    return true;
  }

};

REGISTER_FREE_MEMORY_CALLBACK("garbage_collector_callback", GarbageCollectorCallback)
}