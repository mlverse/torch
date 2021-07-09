#include "torch_types.h"
#include "utils.h"

void* rcpp_call_hook (void* x, void* hook);

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchFunctionPtr> cpp_trace_function (Rcpp::Function fn, XPtrTorchStack inputs,
                        XPtrTorchCompilationUnit compilation_unit,
                        XPtrTorchstring name,
                        bool strict = true, 
                        XPtrTorchScriptModule module = R_NilValue,
                        bool should_mangle = true,
                        bool manage_memory = true
                        )
{
  auto output = XPtrTorchStack(lantern_Stack_new());
  std::function<void*(void*)> r_fn = [&fn, &output](void* inputs) {
    // we don't control this memory as it will be controlled by
    // the lantern side function therefore we use a no-op deleter.
    auto inputs_ = XPtrTorchStack(inputs, [](void* x) {});
    output = Rcpp::as<XPtrTorchStack>(fn(inputs_));
    return output.get();
  };
  
  std::function<void(void*)> deleter;
  if (manage_memory)
  {
    deleter = [] (void* x) {lantern_FunctionPtr_delete(x);};
  }
  else 
  {
    deleter = [](void* x) -> void {};
  }
  
  XPtrTorchTraceableFunction traceable_fn = lantern_create_traceable_fun(&rcpp_call_hook, (void*) &r_fn);
  auto tr_fn = XPtrTorchFunctionPtr(
    lantern_trace_fn(
      traceable_fn.get(), 
      inputs.get(), 
      compilation_unit.get(), 
      strict, 
      module.get(), 
      name.get(), 
      should_mangle
    ), 
    deleter
  );
  
  return make_xptr<XPtrTorchFunctionPtr>(tr_fn);
}

// [[Rcpp::export]]
void cpp_save_traced_fn (Rcpp::XPtr<XPtrTorchFunctionPtr> fn, std::string filename)
{
  lantern_traced_fn_save(fn->get(), filename.c_str());
}

// [[Rcpp::export]]
XPtrTorchCompilationUnit cpp_jit_compilation_unit ()
{
  return XPtrTorchCompilationUnit(lantern_CompilationUnit_new());
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