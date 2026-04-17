#include <iostream>
#define LANTERN_BUILD
#include <c10/core/CPUAllocator.h>
#include <c10/core/impl/alloc_cpu.h>
#include <torch/torch.h>

#include <thread>
#include <unordered_map>

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

std::atomic<bool> lantern_allocator_bypass(false);

// ---- CPU block cache ----
// Caches freed CPU memory blocks keyed by size so that subsequent allocations
// of the same size skip mmap/munmap syscalls. This compensates for R's deferred
// GC: R doesn't free tensors immediately (unlike Python's refcounting), so
// without caching, each op does a fresh mmap while old blocks are still alive.
//
// Parameters (configurable at runtime via lantern API):
//   - min_block_size (default 1KB): don't cache tiny blocks — heap malloc
//     already handles them efficiently, and the hashmap overhead isn't worth it.
//   - max_cache_size (default = GC threshold, typically 4GB): cap on total
//     cached bytes to limit memory retention.
//   - Exact size matching via unordered_map for O(1) lookup.
struct CPUBlockCache {
  std::mutex mtx;
  // size -> list of reusable pointers
  std::unordered_map<size_t, std::vector<void*>> available;
  // ptr -> size (so we know the size on free)
  std::unordered_map<void*, size_t> allocated;
  size_t cached_bytes = 0;
  size_t min_block_size = 1024;         // 1 KB default
  size_t max_cache_size = 4000000000UL; // 4 GB default (overwritten by GC threshold on init)

  void* allocate(size_t nbytes) {
    if (nbytes < min_block_size) return nullptr;
    std::lock_guard<std::mutex> guard(mtx);
    auto it = available.find(nbytes);
    if (it != available.end() && !it->second.empty()) {
      void* ptr = it->second.back();
      it->second.pop_back();
      cached_bytes -= nbytes;
      return ptr;
    }
    return nullptr;
  }

  void free(void* ptr) {
    std::lock_guard<std::mutex> guard(mtx);
    auto it = allocated.find(ptr);
    if (it == allocated.end()) {
      // Not tracked by us, free normally
      c10::free_cpu(ptr);
      return;
    }
    size_t nbytes = it->second;
    if (cached_bytes + nbytes > max_cache_size) {
      // Cache is full, free to OS instead
      c10::free_cpu(ptr);
      allocated.erase(it);
      return;
    }
    available[nbytes].push_back(ptr);
    cached_bytes += nbytes;
  }

  void record(void* ptr, size_t nbytes) {
    if (nbytes < min_block_size) return;
    std::lock_guard<std::mutex> guard(mtx);
    allocated[ptr] = nbytes;
  }

  // Release all cached blocks back to the OS
  void free_cached() {
    std::lock_guard<std::mutex> guard(mtx);
    for (auto& [sz, ptrs] : available) {
      for (void* ptr : ptrs) {
        c10::free_cpu(ptr);
        allocated.erase(ptr);
      }
    }
    available.clear();
    cached_bytes = 0;
  }

  ~CPUBlockCache() {
    for (auto& [sz, ptrs] : available) {
      for (void* ptr : ptrs) {
        c10::free_cpu(ptr);
      }
    }
  }
};

static CPUBlockCache block_cache;

namespace c10 {
struct LanternCPUAllocator final : at::Allocator {
  LanternCPUAllocator() {}
  ~LanternCPUAllocator() override {}
  at::DataPtr allocate(size_t nbytes) override {
    void* data;

    // Try the block cache first (unless caching is disabled)
    if (!lantern_allocator_bypass.load(std::memory_order_relaxed)) {
      data = block_cache.allocate(nbytes);
      if (data) {
        profiledCPUMemoryReporter().New(data, nbytes);
        return {data, data, &CachedDelete, at::Device(at::DeviceType::CPU)};
      }
    }

    {
      // we register every memory allocation
      const std::lock_guard<std::mutex> lock(mtx_allocated);
      allocated_memory += nbytes;
    }

    if (std::this_thread::get_id() == MAIN_THREAD_ID) {
      // if on the main thread we check if we have allocated 4GB of memory
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
        // GC ran finalizers which populated the cache via CachedDelete.
        // Flush the cache now to return memory to the OS.
        block_cache.free_cached();
      }
    }

    try {
      data = alloc_cpu(nbytes);
    } catch (...) {
      // Free cached blocks first, then try GC
      block_cache.free_cached();
      try {
        data = alloc_cpu(nbytes);
      } catch (...) {
        (*call_r_gc)(true);
        wait_for_gc();
        data = alloc_cpu(nbytes);
      }
    }

    if (!lantern_allocator_bypass.load(std::memory_order_relaxed)) {
      block_cache.record(data, nbytes);
    }
    profiledCPUMemoryReporter().New(data, nbytes);
    return {data, data, &CachedDelete, at::Device(at::DeviceType::CPU)};
  }

  // Deleter that caches the block for reuse
  static void CachedDelete(void* ptr) {
    if (!ptr) return;
    profiledCPUMemoryReporter().Delete(ptr);
    block_cache.free(ptr);
  }

  // Deleter that frees immediately (used by bypass mode)
  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    profiledCPUMemoryReporter().Delete(ptr);
    free_cpu(ptr);
  }

  void copy_data(void* dest, const void* src, std::size_t count) const override {
    default_copy_data(dest, src, count);
  }

  at::DeleterFnPtr raw_deleter() const override { return &CachedDelete; }
};

}  // namespace c10

auto lantern_allocator = at::LanternCPUAllocator();

void _set_lantern_allocator(void (*r_gc)(bool), uint64_t threshold_mb) {
  _lantern_set_call_r_gc(r_gc);
  threshold_call_gc = threshold_mb * 1e6;
  // Default max_cache_size to match GC threshold
  block_cache.max_cache_size = threshold_call_gc;
  c10::SetCPUAllocator(&lantern_allocator, 1);
}

void _lantern_set_allocator_bypass(bool bypass) {
  lantern_allocator_bypass.store(bypass);
}

void _lantern_set_cache_max_size(uint64_t max_size_mb) {
  block_cache.max_cache_size = max_size_mb * 1000000UL;
}

void _lantern_set_cache_min_block_size(uint64_t min_size_bytes) {
  block_cache.min_block_size = min_size_bytes;
}

void _lantern_flush_cache() {
  block_cache.free_cached();
}

double cuda_allocator_reserved_rate;
double cuda_allocator_allocated_rate;
double cuda_allocator_allocated_reserved_rate;

void _lantern_set_cuda_allocator_thresholds (double reserved_rate, double allocated_rate,
                                             double allocated_reserved_rate) {
  cuda_allocator_reserved_rate = reserved_rate;
  cuda_allocator_allocated_rate = allocated_rate;
  cuda_allocator_allocated_reserved_rate = allocated_reserved_rate;
}
