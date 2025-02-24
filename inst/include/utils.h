#include <torch.h>

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <stack>
#include <type_traits>
#include <utility>

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
  ~EventLoop() {
    std::unique_lock<std::mutex> lock(mtx_);
    std::packaged_task<T()> fn;
    while (!tasks_.empty()){
      fn = std::move(tasks_.front());
      tasks_.pop_front();
      if (fn.valid()) {
        fn();  
      }
    }
  }
  void run() {
    while (true) {
      std::packaged_task<T()> fn;
      {
        std::unique_lock<std::mutex> lock(mtx_);
        if (tasks_.empty()) {
          cv_.wait(lock, [this] { return stop_requested_ || !tasks_.empty(); });
        }
        
        if (stop_requested_ && tasks_.empty()) {
          return;
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
  void stop() {
    stop_requested_ = true;
    cv_.notify_all();
  }

 private:
  std::mutex mtx_;
  std::condition_variable cv_;
  std::atomic<bool> stop_requested_{false};
  std::deque<std::packaged_task<T()>> tasks_;
};

template<typename T>
class ThreadPool {
public:
  EventLoop<T> event_loop;
  std::vector<std::thread> threads;

  ThreadPool (int n_threads = 5) {
    for(int i = 0; i < n_threads; i++) {
      threads.push_back(std::thread([this] () {
        this->event_loop.run();
      }));
    }
  }
  void push (std::packaged_task<T()>&& task) {
    this->event_loop.schedule(std::move(task));
  }
  void stop() {
    event_loop.stop();
    for(auto& thread : threads) {
      thread.join();
    }
    threads.clear();
  }
  ~ThreadPool() {
    stop();
  }
};

template <class type>
Rcpp::XPtr<type> make_xptr(type x) {
  auto* out = new type(x);
  return Rcpp::XPtr<type>(out);
}

template <class type>
Rcpp::XPtr<type> make_xptr(type x, std::string dyn_type) {
  auto* out = new type(x);
  auto ptr = Rcpp::XPtr<type>(out);
  ptr.attr("dynamic_type") = dyn_type;
  return ptr;
}

template <class type, int n>
std::array<type, n> std_vector_to_std_array(std::vector<type> x) {
  std::array<type, n> out;
  std::copy_n(x.begin(), n, out.begin());
  return out;
}

XPtrTorchTensor cpp_tensor_undefined();
XPtrTorchTensor to_index_tensor(XPtrTorchTensor t);
torch::Tensor torch_tensor_cpp(
    SEXP x, Rcpp::Nullable<torch::Dtype> dtype = R_NilValue,
    Rcpp::Nullable<torch::Device> device = R_NilValue,
    bool requires_grad = false, bool pin_memory = false);

std::thread::id main_thread_id() noexcept;
