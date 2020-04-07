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
std::deque<std::packaged_task<void()>> backward_tasks;
std::atomic<bool> is_nested;
std::mutex tasks_mutex;
std::mutex backward_tasks_mutex;

void event_loop_thread(std::atomic<bool> &event_loop_running)
{
  Rcpp::Rcout << "entering the event loop thread!" << std::endl;
  Rcpp::Rcout << "event pool running: " << event_loop_running << std::endl;
  
  while (event_loop_running) {
    // process messages
    {
      std::unique_lock<std::mutex> lock(tasks_mutex);
      //std::cout << "number of tasks " << tasks.size() << std::endl;
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

// [[Rcpp::export]]
void cpp_torch_method_backward_self_Tensor (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::XPtr<XPtrTorchTensor> gradient, bool keep_graph, bool create_graph) {
  
  auto self_ptr = self->get();
  auto gradient_ptr = gradient->get();
  auto keep_graph_val = keep_graph;
  auto create_graph_val = create_graph;
  
  std::atomic<bool> event_loop_running;
  
  event_loop_running = true;
  std::future<void> result;
  if (!is_nested)
  {
    Rcpp::Rcout << "spining new thread" << std::endl;
    
    result = std::async([&](){
      std::cout << "running this fun async" << std::endl;
      try
      {
        lantern_Tensor_backward_tensor_tensor_bool_bool(
          self_ptr, gradient_ptr, 
          reinterpret_cast<void*>(&keep_graph_val), 
          reinterpret_cast<void*>(&create_graph_val)
        );
        std::cout << "finished backwarding" << std::endl;
      }
      catch (...)
      {
        event_loop_running = false;
        is_nested = false;
        throw;
      }
      
      event_loop_running = false;
      is_nested = false;
    }); 
    is_nested = true;
    
    Rcpp::Rcout << "thread started!" << std::endl;
  } else {
    
    std::cout << "Creatig new backward task" << std::endl;
    
    std::packaged_task<void()> task([&](){
      std::cout << "running this fun async" << std::endl;
      try
      {
        lantern_Tensor_backward_tensor_tensor_bool_bool(
          self_ptr, gradient_ptr, 
          reinterpret_cast<void*>(&keep_graph_val), 
          reinterpret_cast<void*>(&create_graph_val)
        );
        std::cout << "finished backwarding" << std::endl;
      }
      catch (...)
      {
        event_loop_running = false;
        is_nested = false;
        throw;
      }
      
      event_loop_running = false;
      is_nested = false;
      
    });
    result = task.get_future();
    
    
    {
      std::lock_guard<std::mutex> lock(tasks_mutex);
      std::cout << "Pushing new task" << std::endl;
      backward_tasks.push_front(std::move(task));
    }
    
    std::cout << "Finished publishing task" << std::endl;
    
  }
  event_loop_thread(event_loop_running);
  result.get();
}

void*  rcpp_call_hook (void* x, void* hook) {
  std::cout << "Calling hook" << std::endl;
  return (*reinterpret_cast<std::function<void*(void*)> *>(hook))(x);
}

#include <fstream>
#include <iostream>

// [[Rcpp::export]]
void cpp_tensor_register_hook (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::Function f) {
  Rcpp::Rcout << std::this_thread::get_id() << std::endl;
  
  auto r_hook = (void *)new std::function<void*(void *)>([f](void *x) {
    
    std::cout << "Creating packaged task!" << std::endl;
    
    std::packaged_task<void*()> task([f, x]() {
      auto y = make_xptr<XPtrTorchTensor>(x);
      return Rcpp::as<Rcpp::XPtr<XPtrTorchTensor>>(f(y))->get();
    });
    std::future<void*> result = task.get_future();
    
    std::cout << "OK! Locking tasks to push new task." << std::endl;
    
    {
      std::lock_guard<std::mutex> lock(tasks_mutex);
      std::cout << "Pushing new task" << std::endl;
      tasks.push_front(std::move(task));
    }
    
    std::cout << "Tasked has ben pushed; wait for results" << std::endl;
    
    std::future_status status;
    do {
      status = result.wait_for(std::chrono::seconds(1));
      if (status == std::future_status::timeout) {
        
        std::unique_lock<std::mutex> lock(backward_tasks_mutex);
        //std::cout << "number of tasks " << tasks.size() << std::endl;
        while (!backward_tasks.empty()) {
          auto task(std::move(backward_tasks.front()));
          backward_tasks.pop_front();
          
          // unlock during the task
          lock.unlock();
          std::cout << "running backward task" << std::endl;
          task();
          std::cout << "finished backward task" << std::endl;
          lock.lock();
        }
        
        std::cout << "timeout\n";
      } 
    } while (status != std::future_status::ready); 
    
    // wait on result
    return result.get();
  });
  auto hook = lantern_new_hook(&rcpp_call_hook, r_hook);
  lantern_Tensor_register_hook(self->get(), hook);
}


