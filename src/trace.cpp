#include "torch_types.h"
#include "utils.h"

// [[Rcpp::export]]
int cpp_trace_function (Rcpp::Function fn, Rcpp::XPtr<XPtrTorchStack> inputs,
                        Rcpp::XPtr<XPtrTorchCompilationUnit> compilation_unit)
{
  
  std::function<void*(void*)> r_fn = [&fn](void* inputs) {
    auto inputs_ = make_xptr<XPtrTorchStack>(inputs);
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchStack>>(fn(inputs_));
    return out->get();
  };
  
  XPtrTorch tr_fn = lantern_create_traceable_fun((void*) &r_fn);
  return lantern_trace_fn(tr_fn.get(), inputs->get(), compilation_unit->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchCompilationUnit> cpp_jit_compilation_unit ()
{
  return make_xptr<XPtrTorchCompilationUnit>(lantern_CompilationUnit_new());
}
