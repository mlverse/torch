#include <iostream>
#define LANTERN_BUILD
#include "lantern/lantern.h"
#include <torch/torch.h>
#include <thread>
#include <c10/core/CPUAllocator.h>
#include "utils.hpp"

const std::thread::id MAIN_THREAD_ID = std::this_thread::get_id();

namespace c10 
{
struct LanternCPUAllocator final : at::Allocator {
  LanternCPUAllocator() {}
  ~LanternCPUAllocator() override {}
  at::DataPtr allocate(size_t nbytes) const override {
    
    void* data;
    if (std::this_thread::get_id() == MAIN_THREAD_ID) 
    {
        try
        {
            // try first allocation
            data = alloc_cpu(nbytes);
        }
        catch(...)
        {
            // we are going to call R gc here!
            std::cout << "allocating " << nbytes << "bytes in the theread id " << 
            std::this_thread::get_id() << std::endl;
            std::cout << "we are going to call the R garbage collector now!" << std::endl;
            
            // then try allocating again!
            data = alloc_cpu(nbytes);
        }
                
    }
    else
    {
        // if not on main thread we don't have any custom behavior
        // because we won't be able to call the R garbage collector
        // anyway.
        data = alloc_cpu(nbytes);
    }

    profiledCPUMemoryReporter().New(data, nbytes);
    return {data, data, &ReportAndDelete, at::Device(at::DeviceType::CPU)};
  }

  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    profiledCPUMemoryReporter().Delete(ptr);
    free_cpu(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }
};

} // namespace c10

auto lantern_allocator = at::LanternCPUAllocator();

void _set_lantern_allocator ()
{
    c10::SetCPUAllocator(&lantern_allocator, 1);
}