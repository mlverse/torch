#include <torch.h>

// [[Rcpp::export]]
XPtrTorchjit_named_parameter_list cpp_jit_script_module_parameters(
    XPtrTorchScriptModule self, bool recurse) {
  return XPtrTorchjit_named_parameter_list(
      lantern_ScriptModule_parameters(self.get(), recurse));
}

// [[Rcpp::export]]
XPtrTorchjit_named_buffer_list cpp_jit_script_module_buffers(
    XPtrTorchScriptModule self, bool recurse) {
  return XPtrTorchjit_named_buffer_list(
      lantern_ScriptModule_buffers(self.get(), recurse));
}

// [[Rcpp::export]]
void cpp_jit_script_module_train(XPtrTorchScriptModule self, bool on) {
  _lantern_ScriptModule_train(self.get(), on);
}

// [[Rcpp::export]]
void cpp_jit_script_module_set_optimized(XPtrTorchScriptModule self, bool on) {
  _lantern_ScriptModule_set_optimized(self.get(), on);
}

// [[Rcpp::export]]
bool cpp_jit_script_module_is_training(XPtrTorchScriptModule self) {
  return _lantern_ScriptModule_is_training(self.get());
}

// [[Rcpp::export]]
bool cpp_jit_script_module_is_optimized(XPtrTorchScriptModule self) {
  return _lantern_ScriptModule_is_optimized(self.get());
}

// [[Rcpp::export]]
void cpp_jit_script_module_register_parameter(XPtrTorchScriptModule self,
                                              XPtrTorchstring name,
                                              XPtrTorchTensor v,
                                              bool is_buffer) {
  lantern_ScriptModule_register_parameter(self.get(), name.get(), v.get(),
                                          is_buffer);
}

// [[Rcpp::export]]
void cpp_jit_script_module_register_buffer(XPtrTorchScriptModule self,
                                           XPtrTorchstring name,
                                           XPtrTorchTensor v) {
  lantern_ScriptModule_register_buffer(self.get(), name.get(), v.get());
}

// [[Rcpp::export]]
void cpp_jit_script_module_register_module(XPtrTorchScriptModule self,
                                           XPtrTorchstring name,
                                           XPtrTorchScriptModule module) {
  lantern_ScriptModule_register_module(self.get(), name.get(), module.get());
}

// [[Rcpp::export]]
void cpp_jit_script_module_to(XPtrTorchScriptModule self,
                              XPtrTorchDevice device, bool non_blocking) {
  lantern_ScriptModule_to(self.get(), device.get(), non_blocking);
}

// [[Rcpp::export]]
XPtrTorchjit_named_module_list cpp_jit_script_module_modules(
    XPtrTorchScriptModule self) {
  return XPtrTorchjit_named_module_list(
      lantern_ScriptModule_modules(self.get()));
}

// [[Rcpp::export]]
XPtrTorchjit_named_module_list cpp_jit_script_module_children(
    XPtrTorchScriptModule self) {
  return XPtrTorchjit_named_module_list(
      lantern_ScriptModule_children(self.get()));
}

// [[Rcpp::export]]
XPtrTorchScriptMethod cpp_jit_script_module_find_method(
    XPtrTorchScriptModule self, XPtrTorchstring basename) {
  return XPtrTorchScriptMethod(
      lantern_ScriptModule_find_method(self.get(), basename.get()));
}

// [[Rcpp::export]]
XPtrTorchStack cpp_jit_script_method_call(XPtrTorchScriptMethod self,
                                          XPtrTorchStack inputs) {
  return XPtrTorchStack(lantern_ScriptMethod_call(self.get(), inputs.get()));
}

// [[Rcpp::export]]
SEXP cpp_jit_script_method_graph_print(XPtrTorchScriptMethod self) {
  return XPtrTorchstring(lantern_ScriptMethod_graph_print(self.get()));
}

// [[Rcpp::export]]
SEXP cpp_jit_last_executed_optimized_graph_print() {
  return XPtrTorchstring(lantern_last_executed_optimized_graph_print());
}

// [[Rcpp::export]]
XPtrTorchScriptModule cpp_jit_script_module_new(XPtrTorchCompilationUnit cu,
                                                XPtrTorchstring name) {
  return XPtrTorchScriptModule(lantern_ScriptModule_new(cu.get(), name.get()));
}

// [[Rcpp::export]]
void cpp_jit_script_module_add_constant(XPtrTorchScriptModule self,
                                        XPtrTorchstring name,
                                        XPtrTorchIValue value) {
  lantern_ScriptModule_add_constant(self.get(), name.get(), value.get());
}

// [[Rcpp::export]]
void cpp_jit_script_module_add_method(XPtrTorchScriptModule self,
                                      Rcpp::XPtr<XPtrTorch> method) {
  lantern_ScriptModule_add_method(self.get(), method->get());
}

// [[Rcpp::export]]
SEXP cpp_jit_script_module_find_constant(XPtrTorchScriptModule self,
                                         XPtrTorchstring name) {
  void* ret = lantern_ScriptModule_find_constant(self.get(), name.get());
  if (ret) {
    return XPtrTorchIValue(ret);
  } else {
    return R_NilValue;
  }
}

// [[Rcpp::export]]
void cpp_jit_script_module_save(XPtrTorchScriptModule self,
                                XPtrTorchstring path) {
  lantern_ScriptModule_save(self.get(), path.get());
}

// [[Rcpp::export]]
void cpp_jit_script_module_save_for_mobile(XPtrTorchScriptModule self,
                                           XPtrTorchstring path) {
  lantern_ScriptModule_save_for_mobile(self.get(), path.get());
}
