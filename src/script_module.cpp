#include "torch_types.h"
#include "utils.h"

// [[Rcpp::export]]
XPtrTorchjit_named_parameter_list cpp_jit_script_module_parameters (XPtrTorchScriptModule self)
{
  return XPtrTorchjit_named_parameter_list(lantern_ScriptModule_parameters(self.get()));
}