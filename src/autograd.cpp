#include <torch.h>

#include <deque>
#include <future>
#include <thread>

#define LANTERN_CALLBACK_START try {
#define LANTERN_CALLBACK_END(unknown, ret) \
  }                                        \
  catch (const std::exception& ex) {       \
    lanternSetLastError(ex.what());        \
    return (void*)(ret);                   \
  }                                        \
  catch (std::string & ex) {               \
    lanternSetLastError(ex.c_str());       \
    return (void*)(ret);                   \
  }                                        \
  catch (...) {                            \
    lanternSetLastError(unknown);          \
    return (void*)(ret);                   \
  }

// [[Rcpp::export]]
void cpp_autograd_set_grad_mode(bool enabled) {
  lantern_autograd_set_grad_mode(enabled);
}

// [[Rcpp::export]]
bool cpp_autograd_is_enabled() { return lantern_autograd_is_enabled(); }

// [[Rcpp::export]]
void cpp_autograd_set_detect_anomaly(bool enabled) {
  lantern_autograd_set_detect_anomaly(enabled);
}

// [[Rcpp::export]]
bool cpp_autograd_detect_anomaly_is_enabled() {
  return lantern_autograd_detect_anomaly_is_enabled();
}

// [[Rcpp::export]]
torch::Tensor cpp_tensor_grad(torch::Tensor self) {
  return torch::Tensor(lantern_Tensor_grad(self.get()));
}

// [[Rcpp::export]]
void cpp_tensor_set_grad_(torch::Tensor self, torch::Tensor new_grad) {
  lantern_Tensor_set_grad_(self.get(), new_grad.get());
}

// [[Rcpp::export]]
bool cpp_tensor_requires_grad(torch::Tensor self) {
  return lantern_Tensor_requires_grad(self.get());
}

namespace {

EventLoop<void*> gTasks;
EventLoop<void> gBackwardTasks;

void schedule_backward_task(std::packaged_task<void()>&& task) {
  if (std::this_thread::get_id() == main_thread_id()) {
    // NOTE: pre-C++-14 workaround for "moving" `task` into a lambda, not pretty
    auto const task_sp =
        std::make_shared<std::packaged_task<void()>>(std::move(task));

    auto const thr_sp = std::make_shared<std::thread>();
    *thr_sp = std::thread([task_sp, thr_sp] {
      auto thr_join_sg = makeScopeGuard([thr_sp] {
        gTasks.schedule(std::packaged_task<void*()>([thr_sp]() -> void* {
          thr_sp->join();
          return nullptr;
        }));
      });
      (*task_sp)();
    });
  } else {
    gBackwardTasks.schedule(std::move(task));
  }
}

}  // namespace

// [[Rcpp::export]]
void cpp_torch_method__backward_self_Tensor_inputs_TensorList(
    torch::Tensor self, torch::TensorList inputs,
    torch::optional::Tensor gradient, torch::optional::bool_t retain_graph,
    torch::bool_t create_graph) {
  std::function<void()> backward([&]() {
    auto sg = makeScopeGuard([] { gTasks.stopWhenEmpty(); });

    lantern_Tensor__backward_tensor_tensorlist_tensor_bool_bool(
        self.get(), inputs.get(), gradient.get(), retain_graph.get(),
        create_graph.get());
  });

  std::packaged_task<void()> task(backward);
  auto result_fut = task.get_future();

  schedule_backward_task(std::move(task));
  gTasks.run();

  result_fut.get();
}

// [[Rcpp::export]]
void cpp_autograd_backward(Rcpp::XPtr<XPtrTorchvariable_list> tensors,
                           Rcpp::XPtr<XPtrTorchvariable_list> grad_tensors,
                           bool retain_graph, bool create_graph) {
  auto tensors_ = tensors->get();
  auto grad_tensors_ = grad_tensors->get();

  std::function<void()> backward([&]() {
    auto sg = makeScopeGuard([] { gTasks.stopWhenEmpty(); });

    lantern_autograd_backward(tensors_, grad_tensors_, retain_graph,
                              create_graph);
  });

  std::packaged_task<void()> task(backward);
  auto result_fut = task.get_future();
  schedule_backward_task(std::move(task));
  gTasks.run();

  result_fut.get();
}

void* rcpp_call_hook(void* x, void* hook) {
  return (*reinterpret_cast<std::function<void*(void*)>*>(hook))(x);
}

// Since hooks are arbitrary R functions, they must run in the main
// thread. In order to do it we call `backward` in a different thread
// see `cpp_backward` and leave the main thread free to execute the R
// calbacks.
//
// However, the R hooks can by themselves call `backward` again and torch
// does not allow us to use a different thread. so while we are waiting for
// the hook to finish, we allow this thread to execute tasks in the thread
// the hook as been called.
//
// [[Rcpp::export]]
unsigned int cpp_tensor_register_hook(Rcpp::XPtr<XPtrTorchTensor> self,
                                      Rcpp::Function f) {
  auto r_hook = (void*)new std::function<void*(void*)>([f](void* x) {
    std::packaged_task<void*()> task([f, x]() {
      LANTERN_CALLBACK_START
      SEXP y = PROTECT(XPtrTorchTensor(x));
      auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchTensor>>(f(y))->get();
      UNPROTECT(1);
      return out;
      LANTERN_CALLBACK_END("Unknon error in hook.", NULL)
    });
    std::future<void*> result_fut = task.get_future();

    gTasks.schedule(std::move(task));

    std::future_status status;
    do {
      status = result_fut.wait_for(std::chrono::seconds(0));
      if (status == std::future_status::timeout) {
        gBackwardTasks.stopWhenEmpty();
        gBackwardTasks.run();
      }
      std::this_thread::yield();
    } while (status != std::future_status::ready);

    return result_fut.get();
  });

  auto hook = lantern_new_hook(&rcpp_call_hook, r_hook);
  return lantern_Tensor_register_hook(self->get(), hook);
}

// [[Rcpp::export]]
void cpp_tensor_remove_hook(Rcpp::XPtr<XPtrTorchTensor> self,
                            unsigned int pos) {
  lantern_Tensor_remove_hook(self->get(), pos);
}

void* rcpp_call_forward(void* forward, void* ctx, void* inputs) {
  return (*reinterpret_cast<std::function<void*(void*, void*)>*>(forward))(
      ctx, inputs);
}

void rcpp_delete_variable_list(void* x) {
  delete reinterpret_cast<torch::variable_list*>(x);
}

void* rcpp_variable_list_ptr(void* x) {
  return reinterpret_cast<torch::variable_list*>(x)->get();
}

void rcpp_delete_lambda_fun(void* x) {
  delete reinterpret_cast<std::function<void*(void*, void*)>*>(x);
}

// [[Rcpp::export]]
void register_lambda_function_deleter() {
  set_delete_lambda_fun(&rcpp_delete_lambda_fun);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_Function_lambda(Rcpp::Function f) {
  // This function is deleted once the LanternNode holding it or in the of
  // forward, right after it has been called.
  // See the LanternFunction::apply code for the deleter.
  auto fun = new std::function<void*(void*, void*)>([f](void* ctx,
                                                        void* inputs) {
    LANTERN_CALLBACK_START

    std::packaged_task<void*()> task([f, ctx, inputs]() {
      auto inp = XPtrTorchvariable_list(inputs);
      auto con = make_xptr<XPtrTorch>(ctx);
      auto r_out = f(con, inp);
      // A deleter will be called in the Lantern side to make sure this pointer
      // gets correctly deleted.
      auto output =
          new torch::variable_list(Rcpp::as<torch::variable_list>(r_out));
      return (void*)output;
    });

    auto result_fut = task.get_future();

    gTasks.schedule(std::move(task));

    return result_fut.get();
    LANTERN_CALLBACK_END("Unknown error in lambda function.",
                         new torch::variable_list(lantern_variable_list_new()))
  });

  auto out = XPtrTorch(lantern_Function_lambda(&rcpp_call_forward, (void*)fun,
                                               &rcpp_delete_variable_list,
                                               &rcpp_variable_list_ptr));
  return make_xptr<XPtrTorch>(out);
}

// [[Rcpp::export]]
torch::variable_list cpp_Function_apply(torch::variable_list inputs,
                                        Rcpp::XPtr<XPtrTorch> forward,
                                        Rcpp::XPtr<XPtrTorch> backward) {
  std::function<XPtrTorchvariable_list()> apply([&]() {
    auto sg = makeScopeGuard([] { gTasks.stopWhenEmpty(); });

    try {
      return XPtrTorchvariable_list(lantern_Function_apply(
          inputs.get(), forward->get(), backward->get()));
    } catch (std::string& ex) {
      throw Rcpp::exception(ex.c_str());
    } catch (const std::exception& ex) {
      throw Rcpp::exception(ex.what());
    }
  });

  std::packaged_task<XPtrTorchvariable_list()> task(apply);
  auto result_fut = task.get_future();

  auto const thr_sp = std::make_shared<std::thread>(std::move(task));
  auto thr_join_sg = makeScopeGuard([thr_sp] { thr_sp->join(); });

  gTasks.run();

  return result_fut.get();
}

// [[Rcpp::export]]
void cpp_autograd_context_save_for_backward(Rcpp::XPtr<XPtrTorch> self,
                                            torch::variable_list vars) {
  lantern_AutogradContext_save_for_backward(self->get(), vars.get());
}

// [[Rcpp::export]]
torch::variable_list cpp_autograd_context_get_saved_variables(
    Rcpp::XPtr<XPtrTorch> self) {
  return XPtrTorchvariable_list(
      lantern_AutogradContext_get_saved_variables(self->get()));
}

// [[Rcpp::export]]
void cpp_autograd_context_set_arguments(Rcpp::XPtr<XPtrTorch> self,
                                        torch::vector::string names,
                                        torch::vector::bool_t needs_grad) {
  lantern_AutogradContext_set_arguments(self->get(), names.get(),
                                        needs_grad.get());
}

// [[Rcpp::export]]
torch::vector::string cpp_autograd_context_get_argument_names(
    Rcpp::XPtr<XPtrTorch> self) {
  return torch::vector::string(
      lantern_AutogradContext_get_argument_names(self->get()));
}

// [[Rcpp::export]]
torch::vector::bool_t cpp_autograd_context_get_argument_needs_grad(
    Rcpp::XPtr<XPtrTorch> self) {
  return torch::vector::bool_t(
      lantern_AutogradContext_get_argument_needs_grad(self->get()));
}

// [[Rcpp::export]]
void cpp_autograd_context_set_saved_variables_names(
    Rcpp::XPtr<XPtrTorch> self, torch::vector::string names) {
  lantern_AutogradContext_set_saved_variables_names(self->get(), names.get());
}

// [[Rcpp::export]]
torch::vector::string cpp_autograd_context_get_saved_variables_names(
    Rcpp::XPtr<XPtrTorch> self) {
  return torch::vector::string(
      lantern_AutogradContext_get_saved_variables_names(self->get()));
}

// [[Rcpp::export]]
void cpp_autograd_context_mark_dirty(
    Rcpp::XPtr<XPtrTorch> self, Rcpp::XPtr<XPtrTorchvariable_list> inputs) {
  lantern_AutogradContext_mark_dirty(self->get(), inputs->get());
}

// [[Rcpp::export]]
void cpp_autograd_context_mark_non_differentiable(
    Rcpp::XPtr<XPtrTorch> self, Rcpp::XPtr<XPtrTorchvariable_list> outputs) {
  lantern_AutogradContext_mark_non_differentiable(self->get(), outputs->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_tensor_grad_fn(Rcpp::XPtr<XPtrTorchTensor> self) {
  return make_xptr<XPtrTorch>(lantern_Tensor_grad_fn(self->get()));
}

// [[Rcpp::export]]
bool cpp_pointer_is_null(Rcpp::XPtr<XPtrTorchTensor> x) {
  if (x->get() == NULL) {
    return true;
  } else {
    return false;
  }
}

// [[Rcpp::export]]
std::string cpp_autograd_node_name(Rcpp::XPtr<XPtrTorch> self) {
  return std::string(lantern_Node_name(self->get()));
}

// [[Rcpp::export]]
Rcpp::List cpp_autograd_node_next_edges(Rcpp::XPtr<XPtrTorch> self) {
  auto next_edges = XPtrTorch(lantern_Node_next_edges(self->get()),
                              lantern_autograd_edge_list_delete);

  Rcpp::List out;
  auto size = lantern_edge_list_size(next_edges.get());
  for (int i = 0; i < size; i++) {
    out.push_back(Rcpp::XPtr<XPtrTorch>(
        new XPtrTorch(lantern_edge_list_at(next_edges.get(), i),
                      lantern_autograd_edge_delete)));
  }

  return out;
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_autograd_edge_function(Rcpp::XPtr<XPtrTorch> self) {
  return make_xptr<XPtrTorch>(lantern_Edge_function(self->get()));
}

// [[Rcpp::export]]
torch::variable_list cpp_autograd_grad(torch::variable_list outputs,
                                       torch::variable_list inputs,
                                       torch::variable_list grad_outputs,
                                       bool retain_graph, bool create_graph,
                                       bool allow_unused) {
  void* out;
  std::function<void()> grad_fn([&]() {
    auto sg = makeScopeGuard([] { gTasks.stopWhenEmpty(); });
    out = lantern_autograd_grad(outputs.get(), inputs.get(), grad_outputs.get(),
                                retain_graph, create_graph, allow_unused);
  });

  std::packaged_task<void()> task(grad_fn);
  auto result_fut = task.get_future();
  schedule_backward_task(std::move(task));
  gTasks.run();
  result_fut.get();
  return torch::variable_list(out);
}
