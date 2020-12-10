#include "torch_types.h"
#include "utils.h"

void* rcpp_call_hook (void* x, void* hook);

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_trace_function (Rcpp::Function fn, Rcpp::XPtr<XPtrTorchStack> inputs,
                        Rcpp::XPtr<XPtrTorchCompilationUnit> compilation_unit)
{
  
  std::function<void*(void*)> r_fn = [&fn](void* inputs) {
    auto inputs_ = make_xptr<XPtrTorchStack>(inputs);
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchStack>>(fn(inputs_));
    return out->get();
  };
  
  XPtrTorch traceable_fn = lantern_create_traceable_fun(&rcpp_call_hook, (void*) &r_fn);
  XPtrTorch tr_fn = lantern_trace_fn(traceable_fn.get(), inputs->get(), compilation_unit->get());
  
  return make_xptr<XPtrTorch>(tr_fn);
}

// [[Rcpp::export]]
void cpp_save_traced_fn (Rcpp::XPtr<XPtrTorch> fn, std::string filename)
{
  lantern_traced_fn_save(fn->get(), filename.c_str());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchCompilationUnit> cpp_jit_compilation_unit ()
{
  return make_xptr<XPtrTorchCompilationUnit>(lantern_CompilationUnit_new());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchStack> cpp_call_traced_fn (Rcpp::XPtr<XPtrTorch> fn, 
                                               Rcpp::XPtr<XPtrTorchStack> inputs)
{
  XPtrTorchStack out = lantern_call_traced_fn(fn->get(), inputs->get());
  return make_xptr<XPtrTorchStack>(out);
}

// [[Rcpp::export]]
std::string cpp_traced_fn_graph_print (Rcpp::XPtr<XPtrTorch> fn)
{
  const char * s = lantern_traced_fn_graph_print(fn->get());
  auto out = std::string(s);
  lantern_const_char_delete(s);
  return out;
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchJITModule> cpp_jit_load (std::string path)
{
  XPtrTorchJITModule out = lantern_jit_load(path.c_str());
  return make_xptr<XPtrTorchJITModule>(out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchStack> cpp_call_jit_script (Rcpp::XPtr<XPtrTorchJITModule> module, 
                                                Rcpp::XPtr<XPtrTorchStack> inputs)
{
  XPtrTorchStack out = lantern_call_jit_script(module->get(), inputs->get());
  return make_xptr<XPtrTorchStack>(out);
}