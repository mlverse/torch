#include "torch_types.h"
#include "utils.h"

// [[Rcpp::export]]
XPtrTorchjit_named_parameter_list cpp_jit_script_module_parameters (XPtrTorchScriptModule self, bool recurse)
{
  return XPtrTorchjit_named_parameter_list(lantern_ScriptModule_parameters(self.get(), recurse));
}

// [[Rcpp::export]]
XPtrTorchjit_named_buffer_list cpp_jit_script_module_buffers (XPtrTorchScriptModule self, bool recurse)
{
  return XPtrTorchjit_named_buffer_list(lantern_ScriptModule_buffers(self.get(), recurse));
}

// [[Rcpp::export]]
void cpp_jit_script_module_train (XPtrTorchScriptModule self, bool on)
{
  _lantern_ScriptModule_train(self.get(), on);
}

// [[Rcpp::export]]
void cpp_jit_script_module_set_optimized (XPtrTorchScriptModule self, bool on)
{
  _lantern_ScriptModule_set_optimized(self.get(), on);
}

// [[Rcpp::export]]
bool cpp_jit_script_module_is_training (XPtrTorchScriptModule self)
{
  return _lantern_ScriptModule_is_training(self.get());
}

// [[Rcpp::export]]
bool cpp_jit_script_module_is_optimized (XPtrTorchScriptModule self)
{
  return _lantern_ScriptModule_is_optimized(self.get());
}

// [[Rcpp::export]]
void cpp_jit_script_module_register_parameter (XPtrTorchScriptModule self, 
                                               XPtrTorchstring name, 
                                               XPtrTorchTensor v,
                                               bool is_buffer)
{
  lantern_ScriptModule_register_parameter(self.get(), name.get(), v.get(), is_buffer);
}

// [[Rcpp::export]]
void cpp_jit_script_module_register_buffer (XPtrTorchScriptModule self,
                                            XPtrTorchstring name,
                                            XPtrTorchTensor v) 
{
  lantern_ScriptModule_register_buffer(self.get(), name.get(), v.get());  
}

// [[Rcpp::export]]
void cpp_jit_script_module_to (XPtrTorchScriptModule self, XPtrTorchDevice device, 
                               bool non_blocking)
{
  lantern_ScriptModule_to(self.get(), device.get(), non_blocking);
}

// [[Rcpp::export]]
XPtrTorchjit_named_module_list cpp_jit_script_module_modules (XPtrTorchScriptModule self)
{
  return XPtrTorchjit_named_module_list(lantern_ScriptModule_modules(self.get()));
}

// [[Rcpp::export]]
XPtrTorchjit_named_module_list cpp_jit_script_module_children (XPtrTorchScriptModule self)
{
  return XPtrTorchjit_named_module_list(lantern_ScriptModule_children(self.get()));
}


