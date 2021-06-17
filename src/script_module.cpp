#include "torch_types.h"
#include "utils.h"

// [[Rcpp::export]]
XPtrTorchjit_named_parameter_list cpp_jit_script_module_parameters (XPtrTorchScriptModule self)
{
  return XPtrTorchjit_named_parameter_list(lantern_ScriptModule_parameters(self.get()));
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
