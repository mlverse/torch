#include "torchr_types.h"
#include "utils.hpp"
#include <deque>
#include <future>
#include <thread>

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

std::deque<std::packaged_task<void*()>> tasks;
std::deque<std::packaged_task<void()>> backward_tasks;
std::atomic<int> n_tasks = {0};
std::mutex tasks_mutex;
std::mutex backward_tasks_mutex;

void event_loop_thread(std::atomic<bool> &event_loop_running)
{
  
  while (event_loop_running) {
    
    {
      std::unique_lock<std::mutex> lock(tasks_mutex);
      while (!tasks.empty()) {
        auto task(std::move(tasks.front()));
        tasks.pop_front();
        
        lock.unlock();
        task();
        lock.lock();
      }
    }
    
  }
  
}

// [[Rcpp::export]]
void cpp_torch_method_backward_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> gradient, bool keep_graph, bool create_graph) {
  
  auto self_ptr = self->get();
  auto gradient_ptr = gradient->get();
  auto keep_graph_val = keep_graph;
  auto create_graph_val = create_graph;
  
  std::atomic<bool> event_loop_running;
  
  event_loop_running = true;
  std::future<void> result;
  
  std::function<void()> backward ([&](){
    
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
      n_tasks = n_tasks - 1;
      throw;
    }
    
    event_loop_running = false;
    n_tasks = n_tasks - 1;
  });
  
  if (n_tasks == 0)
  {
    n_tasks = n_tasks + 1;
    result = std::async(backward); 
  } 
  else 
  {
    n_tasks = n_tasks + 1;
    std::packaged_task<void()> task(backward);
    result = task.get_future();
    
    {
      std::lock_guard<std::mutex> lock(tasks_mutex);
      backward_tasks.push_front(std::move(task));
    }
    
  }
  
  event_loop_thread(event_loop_running);
  result.get();
}

void*  rcpp_call_hook (void* x, void* hook) {
  return (*reinterpret_cast<std::function<void*(void*)> *>(hook))(x);
}

// [[Rcpp::export]]
void cpp_tensor_register_hook (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::Function f) {
  
  auto r_hook = (void *)new std::function<void*(void *)>([f](void *x) {
    
    std::packaged_task<void*()> task([f, x]() {
      auto y = make_xptr<XPtrTorchTensor>(x);
      return Rcpp::as<Rcpp::XPtr<XPtrTorchTensor>>(f(y))->get();
    });
    std::future<void*> result = task.get_future();
    
    {
      std::lock_guard<std::mutex> lock(tasks_mutex);
      tasks.push_front(std::move(task));
    }
    
    std::future_status status;
    do {
      status = result.wait_for(std::chrono::seconds(0));
      if (status == std::future_status::timeout) {
        
        std::unique_lock<std::mutex> lock(backward_tasks_mutex);
        while (!backward_tasks.empty()) {
          auto task(std::move(backward_tasks.front()));
          backward_tasks.pop_front();
          
          // unlock during the task
          lock.unlock();
          task();
          lock.lock();
        }
        
      } 
    } while (status != std::future_status::ready); 
    
    return result.get();
  });
  
  auto hook = lantern_new_hook(&rcpp_call_hook, r_hook);
  lantern_Tensor_register_hook(self->get(), hook);
}


