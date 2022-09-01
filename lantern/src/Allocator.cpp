#include <iostream>
#define LANTERN_BUILD
#include <c10/core/CPUAllocator.h>
#include <c10/core/impl/alloc_cpu.h>
#include <torch/torch.h>

#include <thread>

#include "AllocatorUtils.h"
#include "lantern/lantern.h"
#include "utils.hpp"

const std::thread::id MAIN_THREAD_ID = std::this_thread::get_id();
uint64_t allocated_memory;
uint64_t threshold_call_gc;
std::mutex mtx_allocated;
std::mutex mtx_gc_called;
std::condition_variable cv_gc_called;

void (*call_r_gc)(bool) = nullptr;


EventLoop<void> delete_tasks;

// the R gc must be set whenever liblantern is loaded.
void _lantern_set_call_r_gc(void (*fn)(bool)) { call_r_gc = fn; }
void _lantern_set_gc_called (bool called) { 
  if (called && delete_tasks.is_running) {
    delete_tasks.stopWhenEmpty();  
  }
}

void wait_for_gc () { 
  if (std::this_thread::get_id() != MAIN_THREAD_ID) {
    delete_tasks.run();  
  }
}

namespace c10 {
struct LanternCPUAllocator final : at::Allocator {
  LanternCPUAllocator() {}
  ~LanternCPUAllocator() override {}
  at::DataPtr allocate(size_t nbytes) const override {
    void* data;

    {
      // we register every memory allocation
      const std::lock_guard<std::mutex> lock(mtx_allocated);
      allocated_memory += nbytes;
    }

    if (std::this_thread::get_id() == MAIN_THREAD_ID) {
      // if o the main thread we check if we have allocated 4GB of memory
      // and in this case we call the R garbage collector.
      bool call_gc = false;
      {
        const std::lock_guard<std::mutex> lock(mtx_allocated);
        if (allocated_memory > threshold_call_gc) {
          call_gc = true;
          allocated_memory = 0;
        }
      }

      if (call_gc) {
        (*call_r_gc)(false);
      }
    }

    try {
      // try first allocation
      data = alloc_cpu(nbytes);
    } catch (...) {
      // Use R garbage collector and see if we can
      // allocate more memory.
      (*call_r_gc)(true);
      wait_for_gc();
      // then try allocating again!
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

  at::DeleterFnPtr raw_deleter() const override { return &ReportAndDelete; }
};

}  // namespace c10

auto lantern_allocator = at::LanternCPUAllocator();

void _set_lantern_allocator(void (*r_gc)(bool), uint64_t threshold_mb) {
  _lantern_set_call_r_gc(r_gc);
  threshold_call_gc = threshold_mb * 1e6;
  c10::SetCPUAllocator(&lantern_allocator, 1);
}