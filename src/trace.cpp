#include "torch_types.h"
#include "utils.h"

void* rcpp_call_hook (void* x, void* hook);

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchFunctionPtr> cpp_trace_function (Rcpp::Function fn, XPtrTorchStack inputs,
                        Rcpp::XPtr<XPtrTorchCompilationUnit> compilation_unit,
                        bool strict = true)
{
  
  auto output = XPtrTorchStack(lantern_Stack_new());
  std::function<void*(void*)> r_fn = [&fn, &output](void* inputs) {
    auto inputs_ = XPtrTorchStack(inputs);
    output = Rcpp::as<XPtrTorchStack>(fn(inputs_));
    return output.get();
  };
  
  XPtrTorchTraceableFunction traceable_fn = lantern_create_traceable_fun(&rcpp_call_hook, (void*) &r_fn);
  XPtrTorchFunctionPtr tr_fn = lantern_trace_fn(traceable_fn.get(), inputs.get(), compilation_unit->get(), strict);
  
  return make_xptr<XPtrTorchFunctionPtr>(tr_fn);
}

// [[Rcpp::export]]
void cpp_save_traced_fn (Rcpp::XPtr<XPtrTorchFunctionPtr> fn, std::string filename)
{
  lantern_traced_fn_save(fn->get(), filename.c_str());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchCompilationUnit> cpp_jit_compilation_unit ()
{
  return make_xptr<XPtrTorchCompilationUnit>(lantern_CompilationUnit_new());
}

// [[Rcpp::export]]
XPtrTorchStack cpp_call_traced_fn (Rcpp::XPtr<XPtrTorchFunctionPtr> fn, 
                                   XPtrTorchStack inputs)
{
  XPtrTorchStack out = lantern_call_traced_fn(fn->get(), inputs.get());
  return out;
}

// [[Rcpp::export]]
std::string cpp_traced_fn_graph_print (Rcpp::XPtr<XPtrTorchFunctionPtr> fn)
{
  const char * s = lantern_traced_fn_graph_print(fn->get());
  auto out = std::string(s);
  lantern_const_char_delete(s);
  return out;
}

// [[Rcpp::export]]
XPtrTorchScriptModule cpp_jit_load (std::string path)
{
  return XPtrTorchScriptModule(lantern_jit_load(path.c_str()));
}

// [[Rcpp::export]]
XPtrTorchStack cpp_call_jit_script (Rcpp::XPtr<XPtrTorchJITModule> module, 
                                                XPtrTorchStack inputs)
{
  XPtrTorchStack out = lantern_call_jit_script(module->get(), inputs.get());
  return out;
}