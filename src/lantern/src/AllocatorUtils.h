#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <stack>
#include <type_traits>
#include <utility>

extern const std::thread::id MAIN_THREAD_ID;
extern void (*call_r_gc)(bool);
void wait_for_gc();

extern double cuda_allocator_reserved_rate;
extern double cuda_allocator_allocated_rate;
extern double cuda_allocator_allocated_reserved_rate;

template <typename F>
class ScopeGuard {
public:
  explicit ScopeGuard(F&& f) noexcept : f_(std::forward<F>(f)) {}
  ~ScopeGuard() noexcept { f_(); }
private:
  typename std::decay<F>::type f_;
};

// TODO:
// create a factory function for ScopeGuard that disallows the returned
// ScopeGuard object to be discarded immediately (but we need to do so in a
// portable manner, by using the [[nodiscard]] attribute if possible, or
// whatever its equivalent may be in other compilers that need to build {torch}
// but don't support [[nodiscard]] yet)
template <typename F>
ScopeGuard<F> makeScopeGuard(F&& f) {
  return ScopeGuard<F>(std::forward<F>(f));
}

template <typename T>
class EventLoop {
public:
  EventLoop() = default;
  std::atomic<bool> is_running;
  void run() {
    is_running = true;
    auto is_running_sg = makeScopeGuard([this] {
      this->is_running = false;
    });
    while (true) {
      std::packaged_task<T()> fn;
      {
        std::unique_lock<std::mutex> lock(mtx_);
        if (tasks_.empty()) {
          cv_.wait(lock, [this] { return !tasks_.empty(); });
        }
        fn = std::move(tasks_.front());
        tasks_.pop_front();
      }
      if (!fn.valid()) {
        // signalled to stop the current run
        return;
      }
      fn();
    }
  }
  void schedule(std::packaged_task<T()>&& task) {
    {
      std::unique_lock<std::mutex> lock(mtx_);
      tasks_.emplace_front(std::move(task));
    }
    cv_.notify_one();
  }
  void stopWhenEmpty() {
    {
      std::unique_lock<std::mutex> lock(mtx_);
      tasks_.emplace_back();
    }
    cv_.notify_one();
  }
  
private:
  std::mutex mtx_;
  std::condition_variable cv_;
  std::deque<std::packaged_task<T()>> tasks_;
};
