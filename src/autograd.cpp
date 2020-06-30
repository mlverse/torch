#include "torch_types.h"
#include "utils.hpp"
#include <deque>
#include <future>
#include <thread>

#define LANTERN_ERROR_HANDLE                                                    \
if (lanternLastError() != NULL) {                                               \
  std::string last = lanternLastError();                                        \
  lanternLastErrorClear();                                                      \
  throw Rcpp::exception(last.c_str());                                                  \
} 

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
  
  std::packaged_task<void()> task(backward);
  std::future<void> result = task.get_future();
  
  if (n_tasks == 0)
  {
    n_tasks = n_tasks + 1;
    std::thread td (std::move(task));
    td.detach();
  } 
  else 
  {
    n_tasks = n_tasks + 1;
    
    {
      std::lock_guard<std::mutex> lock(tasks_mutex);
      backward_tasks.push_front(std::move(task));
    }
    
  }
  
  event_loop_thread(event_loop_running);
  result.get();
}

// [[Rcpp::export]]
void cpp_autograd_backward (Rcpp::XPtr<XPtrTorchvariable_list> tensors, 
                            Rcpp::XPtr<XPtrTorchvariable_list> grad_tensors,
                            bool retain_graph,
                            bool create_graph
                            )
{
  auto tensors_ = tensors->get();
  auto grad_tensors_ = grad_tensors->get();
  
  std::atomic<bool> event_loop_running;
  event_loop_running = true;
  
  std::function<void()> backward ([&](){
    
    try
    {
      lantern_autograd_backward(tensors_, grad_tensors_, retain_graph, 
                                create_graph);
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
  
  std::packaged_task<void()> task(backward);
  std::future<void> result = task.get_future();
  
  if (n_tasks == 0)
  {
    n_tasks = n_tasks + 1;
    std::thread td (std::move(task));
    td.detach();
  } 
  else 
  {
    n_tasks = n_tasks + 1;
    
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

// Since hooks are arbitrary R functions, they must run in the main 
// thread. In order to do it we call `backward` in a different thread 
// see `cpp_backward` and leave the main thread free to execute the R
// calbacks that are pushed to the `event_loop_thread` as `packaged_tasks`
// in this function.
//
// However, the R hooks can by themselves call `backward` again and torch
// does not allow us to use a different thread. so while we are waiting for
// the hook to finish, we allow this thread to execute tasks in the thread
// the hook as been called.
//
// [[Rcpp::export]]
unsigned int cpp_tensor_register_hook (Rcpp::XPtr<XPtrTorchTensor> self, Rcpp::Function f) {
  
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
  return lantern_Tensor_register_hook(self->get(), hook);
}

// [[Rcpp::export]]
void cpp_tensor_remove_hook (Rcpp::XPtr<XPtrTorchTensor> self, unsigned int pos)
{
  lantern_Tensor_remove_hook(self->get(), pos);
}

void*  rcpp_call_forward (void* forward, void* ctx, void* inputs) {
  return (*reinterpret_cast<std::function<void*(void*, void*)> *>(forward))(ctx, inputs);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_Function_lambda (Rcpp::Function f)
{
  auto fun = (void *)new std::function<void*(void *, void*)>([f](void *ctx, void* inputs) {
    
    std::packaged_task<void*()> task([f, ctx, inputs]() {
      auto inp = make_xptr<XPtrTorchvariable_list>(inputs);
      auto con = make_xptr<XPtrTorch>(ctx);
      auto r_out = f(con, inp);
      auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchvariable_list>>(r_out)->get();
      return out;
    });
    
    std::future<void*> result = task.get_future();
    
    {
      std::lock_guard<std::mutex> lock(tasks_mutex);
      tasks.push_front(std::move(task));
    }
    
    auto out = result.get();
    return out;
  });
  
  XPtrTorch out = lantern_Function_lambda(&rcpp_call_forward, fun);
  return make_xptr<XPtrTorch>(out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchvariable_list> cpp_Function_apply (Rcpp::XPtr<XPtrTorchvariable_list> inputs,
                                                       Rcpp::XPtr<XPtrTorch> forward,
                                                       Rcpp::XPtr<XPtrTorch> backward)
{
  
  auto inputs_ = inputs->get();
  auto forward_ = forward->get();
  auto backward_ = backward->get();
  
  std::atomic<bool> event_loop_running;
  event_loop_running = true;
  
  std::function<XPtrTorchvariable_list()> apply ([&](){
    
    XPtrTorchvariable_list out = nullptr;
    
    try 
    {
      out = lantern_Function_apply(
        inputs_,
        forward_,
        backward_
      );
      LANTERN_ERROR_HANDLE
    }
    catch (const std::exception& ex)
    {
      event_loop_running = false;
      throw Rcpp::exception(ex.what());
    }
    catch (...)
    {
      event_loop_running = false;
      throw;
    }
    
    event_loop_running = false;
    return out;
  });
  
  std::packaged_task<XPtrTorchvariable_list()> task(apply);
  std::future<XPtrTorchvariable_list> result = task.get_future();
  
  std::thread td (std::move(task));
  td.detach();
  
  event_loop_thread(event_loop_running);
  
  auto out = result.get();
  return make_xptr<XPtrTorchvariable_list>(out);
}

// [[Rcpp::export]]
void cpp_autograd_context_save_for_backward (Rcpp::XPtr<XPtrTorch> self, 
                                                              Rcpp::XPtr<XPtrTorchvariable_list> vars)
{
  lantern_AutogradContext_save_for_backward(self->get(), vars->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchvariable_list> cpp_autograd_context_get_saved_variables (Rcpp::XPtr<XPtrTorch> self)
{
  XPtrTorchvariable_list out = lantern_AutogradContext_get_saved_variables(self->get());
  return make_xptr<XPtrTorchvariable_list>(out);
}

// [[Rcpp::export]]
void cpp_autograd_context_set_arguments (Rcpp::XPtr<XPtrTorch> self, std::vector<std::string> names, std::vector<bool> needs_grad)
{
  auto names_ = lantern_vector_string_new();
  for (int i = 0; i < names.size(); i ++)
  {
    lantern_vector_string_push_back(names_, names.at(i).c_str());
  }
  
  auto needs_grad_ = lantern_vector_bool_new();
  for (int i = 0; i < needs_grad.size(); i++)
  {
    lantern_vector_bool_push_back(needs_grad_, needs_grad.at(i));
  }
  
  lantern_AutogradContext_set_arguments(self->get(), names_, needs_grad_);
}

// [[Rcpp::export]]
std::vector<std::string> cpp_autograd_context_get_argument_names (Rcpp::XPtr<XPtrTorch> self)
{
  auto v = lantern_AutogradContext_get_argument_names(self->get());
  auto size = lantern_vector_string_size(v);
  std::vector<std::string> out;
  for (int i = 0; i < size; i++)
  {
    out.push_back(std::string(lantern_vector_string_at(v, i)));
  }
  return out;
}

// [[Rcpp::export]]
std::vector<bool> cpp_autograd_context_get_argument_needs_grad (Rcpp::XPtr<XPtrTorch> self)
{
  auto v = lantern_AutogradContext_get_argument_needs_grad(self->get());
  auto size = lantern_vector_bool_size(v);
  std::vector<bool> out;
  for (int i = 0; i < size; i++)
  {
    out.push_back(lantern_vector_bool_at(v, i));
  }
  return out;
}

// [[Rcpp::export]]
void cpp_autograd_context_set_saved_variables_names (Rcpp::XPtr<XPtrTorch> self, std::vector<std::string> names)
{
  auto names_ = lantern_vector_string_new();
  for (int i = 0; i < names.size(); i ++)
  {
    lantern_vector_string_push_back(names_, names.at(i).c_str());
  }
  
  lantern_AutogradContext_set_saved_variables_names(self->get(), names_);
}

// [[Rcpp::export]]
std::vector<std::string> cpp_autograd_context_get_saved_variables_names (Rcpp::XPtr<XPtrTorch> self)
{
  auto v = lantern_AutogradContext_get_saved_variables_names(self->get());
  auto size = lantern_vector_string_size(v);
  std::vector<std::string> out;
  for (int i = 0; i < size; i++)
  {
    out.push_back(std::string(lantern_vector_string_at(v, i)));
  }
  return out;
}

// [[Rcpp::export]]
void cpp_autograd_context_mark_dirty (Rcpp::XPtr<XPtrTorch> self, Rcpp::XPtr<XPtrTorchvariable_list> inputs)
{
  lantern_AutogradContext_mark_dirty(self->get(), inputs->get());
}

// [[Rcpp::export]]
void cpp_autograd_context_mark_non_differentiable (Rcpp::XPtr<XPtrTorch> self, Rcpp::XPtr<XPtrTorchvariable_list> outputs)
{
  lantern_AutogradContext_mark_non_differentiable(self->get(), outputs->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_tensor_grad_fn (Rcpp::XPtr<XPtrTorchTensor> self)
{
  return make_xptr<XPtrTorch>(lantern_Tensor_grad_fn(self->get()));
}

// [[Rcpp::export]]
std::string cpp_autograd_node_name (Rcpp::XPtr<XPtrTorch> self)
{
  return std::string(lantern_Node_name(self->get()));
}

// [[Rcpp::export]]
Rcpp::List cpp_autograd_node_next_edges (Rcpp::XPtr<XPtrTorch> self)
{
  auto next_edges = lantern_Node_next_edges(self->get());
  
  Rcpp::List out;
  auto size = lantern_edge_list_size(next_edges);
  for (int i = 0; i < size; i ++)
  {
    out.push_back(make_xptr<XPtrTorch>(lantern_edge_list_at(next_edges, i)));
  }
  
  return out;
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_autograd_edge_function(Rcpp::XPtr<XPtrTorch> self)
{
  return make_xptr<XPtrTorch>(lantern_Edge_function(self->get()));
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchvariable_list> cpp_autograd_grad(Rcpp::XPtr<XPtrTorchvariable_list> outputs,
                                                     Rcpp::XPtr<XPtrTorchvariable_list> inputs,
                                                     Rcpp::XPtr<XPtrTorchvariable_list> grad_outputs,
                                                     bool retain_graph,
                                                     bool create_graph,
                                                     bool allow_unused) {
  XPtrTorchvariable_list out = lantern_autograd_grad(
    outputs->get(),
    inputs->get(),
    grad_outputs->get(),
    retain_graph,
    create_graph,
    allow_unused
  );
  return make_xptr<XPtrTorchvariable_list>(out);
}   