#include <torch.h>

#include <deque>
#include <future>
#include <thread>

#define LANTERN_CALLBACK_START                                     \
  try {
#define LANTERN_CALLBACK_END(unknown, ret)                         \
} catch(const std::exception& ex) {                                \
  lanternSetLastError(ex.what());                                  \
  return (void *)(ret);                                            \
} catch(std::string& ex) {                                         \
  lanternSetLastError(ex.c_str());                                 \
  return (void *)(ret);                                            \
} catch(...) {                                                     \
  lanternSetLastError(unknown);                                    \
  return (void *)(ret);                                            \
}

// [[Rcpp::export]]
void cpp_autograd_set_grad_mode (bool enabled) {
  lantern_autograd_set_grad_mode(enabled);
}

// [[Rcpp::export]]
bool cpp_autograd_is_enabled()
{
  return lantern_autograd_is_enabled();  
}
  
// [[Rcpp::export]]
void cpp_autograd_set_detect_anomaly (bool enabled)
{
  lantern_autograd_set_detect_anomaly(enabled);
}
  
// [[Rcpp::export]]
bool cpp_autograd_detect_anomaly_is_enabled ()
{
  return lantern_autograd_detect_anomaly_is_enabled();
}

// [[Rcpp::export]]
torch::Tensor cpp_tensor_grad (torch::Tensor self) {
  return torch::Tensor(lantern_Tensor_grad(self.get()));
}

// [[Rcpp::export]]
void cpp_tensor_set_grad_ (torch::Tensor self, torch::Tensor new_grad)
{
  lantern_Tensor_set_grad_(self.get(), new_grad.get());
}

// [[Rcpp::export]]
bool cpp_tensor_requires_grad (torch::Tensor self) {
  return lantern_Tensor_requires_grad(self.get());
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
void cpp_torch_method__backward_self_Tensor_inputs_TensorList (torch::Tensor self, torch::TensorList inputs, 
                                                              torch::optional::Tensor gradient, 
                                                              torch::optional::bool_t retain_graph, 
                                                              torch::bool_t create_graph) {
  
  std::atomic<bool> event_loop_running(true);
  std::function<void()> backward ([&](){
    
    try
    {
      lantern_Tensor__backward_tensor_tensorlist_tensor_bool_bool(
        self.get(), inputs.get(), gradient.get(), 
        retain_graph.get(), create_graph.get());
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
  
  std::atomic<bool> event_loop_running(true);
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
      LANTERN_CALLBACK_START
      SEXP y = PROTECT(XPtrTorchTensor(x));
      auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchTensor>>(f(y))->get();
      UNPROTECT(1);
      return out;
      LANTERN_CALLBACK_END("Unknon error in hook.", NULL)
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
  auto fun = new std::function<void*(void *, void*)>([f](void *ctx, void* inputs) {
    LANTERN_CALLBACK_START
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
    
    return result.get();
    LANTERN_CALLBACK_END("Unknown error in lambda function.", _lantern_variable_list_new())
  });
  
  auto deleter = [&fun] (void* x) {
    lantern_Function_lambda_delete(x);
    // we should delete the `fun` pointer when the object that refers to it gets
    // deleted.
    delete fun; 
  };
  
  auto out = XPtrTorch(lantern_Function_lambda(&rcpp_call_forward, (void*)fun), deleter);
  return make_xptr<XPtrTorch>(out);
}

// [[Rcpp::export]]
torch::variable_list cpp_Function_apply (torch::variable_list inputs,
                                         Rcpp::XPtr<XPtrTorch> forward,
                                         Rcpp::XPtr<XPtrTorch> backward)
{
  auto forward_ = forward->get();
  auto backward_ = backward->get();
  
  std::atomic<bool> event_loop_running;
  event_loop_running = true;
  
  std::function<XPtrTorchvariable_list()> apply ([&](){
    
    auto out = XPtrTorchvariable_list((void*)nullptr);
    
    try 
    {
      out = XPtrTorchvariable_list(lantern_Function_apply(
        inputs.get(),
        forward_,
        backward_
      ));
    }
    catch(std::string& ex)
    {
      event_loop_running = false;
      throw Rcpp::exception(ex.c_str());
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
  
  return result.get();;
}

// [[Rcpp::export]]
void cpp_autograd_context_save_for_backward (Rcpp::XPtr<XPtrTorch> self, 
                                             torch::variable_list vars)
{
  lantern_AutogradContext_save_for_backward(self->get(), vars.get());
}

// [[Rcpp::export]]
torch::variable_list cpp_autograd_context_get_saved_variables (Rcpp::XPtr<XPtrTorch> self)
{
  return XPtrTorchvariable_list(lantern_AutogradContext_get_saved_variables(self->get()));
}

// [[Rcpp::export]]
void cpp_autograd_context_set_arguments (Rcpp::XPtr<XPtrTorch> self, torch::vector::string names, torch::vector::bool_t needs_grad)
{
  lantern_AutogradContext_set_arguments(self->get(), names.get(), needs_grad.get());
}

// [[Rcpp::export]]
torch::vector::string cpp_autograd_context_get_argument_names (Rcpp::XPtr<XPtrTorch> self)
{
  return torch::vector::string(lantern_AutogradContext_get_argument_names(self->get()));
}

// [[Rcpp::export]]
torch::vector::bool_t cpp_autograd_context_get_argument_needs_grad (Rcpp::XPtr<XPtrTorch> self)
{
  return torch::vector::bool_t(lantern_AutogradContext_get_argument_needs_grad(self->get()));
}

// [[Rcpp::export]]
void cpp_autograd_context_set_saved_variables_names (Rcpp::XPtr<XPtrTorch> self, torch::vector::string names)
{
  lantern_AutogradContext_set_saved_variables_names(self->get(), names.get());
}

// [[Rcpp::export]]
torch::vector::string cpp_autograd_context_get_saved_variables_names (Rcpp::XPtr<XPtrTorch> self)
{
  return torch::vector::string(lantern_AutogradContext_get_saved_variables_names(self->get()));
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
bool cpp_pointer_is_null (Rcpp::XPtr<XPtrTorchTensor> x)
{
  if (x->get() == NULL) {
    return true;
  } else {
    return false;
  }
}

// [[Rcpp::export]]
std::string cpp_autograd_node_name (Rcpp::XPtr<XPtrTorch> self)
{
  return std::string(lantern_Node_name(self->get()));
}

// [[Rcpp::export]]
Rcpp::List cpp_autograd_node_next_edges (Rcpp::XPtr<XPtrTorch> self)
{
  auto next_edges = XPtrTorch(lantern_Node_next_edges(self->get()), 
                              lantern_autograd_edge_list_delete);
  
  Rcpp::List out;
  auto size = lantern_edge_list_size(next_edges.get());
  for (int i = 0; i < size; i ++)
  {
    out.push_back(Rcpp::XPtr<XPtrTorch>(new XPtrTorch(
        lantern_edge_list_at(next_edges.get(), i),
        lantern_autograd_edge_delete
    )));
  }
  
  return out;
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_autograd_edge_function(Rcpp::XPtr<XPtrTorch> self)
{
  return make_xptr<XPtrTorch>(lantern_Edge_function(self->get()));
}

// [[Rcpp::export]]
torch::variable_list cpp_autograd_grad(torch::variable_list outputs,
                                       torch::variable_list inputs,
                                       torch::variable_list grad_outputs,
                                       bool retain_graph,
                                       bool create_graph,
                                       bool allow_unused) {
  XPtrTorchvariable_list out = lantern_autograd_grad(
    outputs.get(),
    inputs.get(),
    grad_outputs.get(),
    retain_graph,
    create_graph,
    allow_unused
  );
  return out;
}   
