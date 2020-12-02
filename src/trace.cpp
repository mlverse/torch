#include "torch_types.h"
#include "utils.h"

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_trace_function (Rcpp::Function fn, Rcpp::XPtr<XPtrTorchStack> inputs,
                        Rcpp::XPtr<XPtrTorchCompilationUnit> compilation_unit)
{
  
  std::function<void*(void*)> r_fn = [&fn](void* inputs) {
    auto inputs_ = make_xptr<XPtrTorchStack>(inputs);
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchStack>>(fn(inputs_));
    return out->get();
  };
  
  XPtrTorch traceable_fn = lantern_create_traceable_fun((void*) &r_fn);
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