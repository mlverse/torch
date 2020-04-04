#include "torchr_types.h"
#include "utils.hpp"
#include <deque>
#include <future>

// [[Rcpp::export]]
void cpp_autograd_set_grad_mode (bool enabled) {
  lantern_autograd_set_grad_mode(enabled);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensor> cpp_tensor_grad (Rcpp::XPtr<XPtrTorchTensor> self) {
  return make_xptr<XPtrTorchTensor>(lantern_Tensor_grad(self->get()));
}

// [[Rcpp::export]]
bool cpp_tensor_requires_grad (Rcpp::XPtr<XPtrTorchTensor> self) {
  return lantern_Tensor_requires_grad(self->get());
}

#include <thread>

std::deque<std::packaged_task<void*()>> tasks;
std::mutex tasks_mutex;
std::atomic<bool> event_loop_running;

void event_loop_thread();

// [[Rcpp::export]]
void cpp_torch_method_backward_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> gradient, bool keep_graph, bool create_graph) {
  
  auto self_ptr = self->get();
  auto gradient_ptr = gradient->get();
  auto keep_graph_val = keep_graph;
  auto create_graph_val = create_graph;
  
  Rcpp::Rcout << "spining new thread" << std::endl;
  
  event_loop_running = true;
  
  std::future<void> result = std::async([&](){
    try
    {
      lantern_Tensor_backward_tensor_tensor_bool_bool(
        self_ptr, gradient_ptr, 
        reinterpret_cast<void*>(&keep_graph_val), 
        reinterpret_cast<void*>(&create_graph_val)
      );  
    }
    catch (...)
    {
      event_loop_running = false;
      throw;
    }
     
    event_loop_running = false;
  });
  
  Rcpp::Rcout << "thread started!" << std::endl;
  
  event_loop_thread();
  
  result.get();
}

void*  rcpp_call_hook (void* x, void* hook) {
  return (*reinterpret_cast<std::function<void*(void*)> *>(hook))(x);
}

// [[Rcpp::export]]
void cpp_tensor_register_hook (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::Function f) {
  Rcpp::Rcout << std::this_thread::get_id() << std::endl;
  
  auto r_hook = (void *)new std::function<void*(void *)>([f](void *x) {
    std::packaged_task<void*()> task([f, x]() {
      auto y = make_xptr<XPtrTorchTensor>(x);
      return Rcpp::as<Rcpp::XPtr<XPtrTorchTensor>>(f(y))->get();
    });
    std::future<void*> result = task.get_future();
    
    {
      std::lock_guard<std::mutex> lock(tasks_mutex);
      tasks.push_back(std::move(task));
    }
    
    // wait on result
    return result.get();
  });
  auto hook = lantern_new_hook(&rcpp_call_hook, r_hook);
  lantern_Tensor_register_hook(self->get(), hook);
}

void event_loop_thread()
{
  Rcpp::Rcout << "entering the event loop thread!" << std::endl;
  Rcpp::Rcout << "event pool running: " << event_loop_running << std::endl;
  
  while (event_loop_running) {
    // process messages
    {
      std::unique_lock<std::mutex> lock(tasks_mutex);
      while (!tasks.empty()) {
        auto task(std::move(tasks.front()));
        tasks.pop_front();
        
        // unlock during the task
        lock.unlock();
        std::cout << "running task" << std::endl;
        task();
        std::cout << "finished task" << std::endl;
        lock.lock();
      }
    }
    
  }
  
  Rcpp::Rcout << "leaving event_lopp thread;" << std::endl;
}
