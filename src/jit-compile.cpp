#include <torch.h>

// [[Rcpp::export]]
XPtrTorchCompilationUnit cpp_jit_compile(XPtrTorchstring source) {
  return XPtrTorchCompilationUnit(lantern_jit_compile(source.get(), nullptr));
}

// [[Rcpp::export]]
XPtrTorchvector_string cpp_jit_compile_list_methods(
    XPtrTorchCompilationUnit cu) {
  return XPtrTorchvector_string(lantern_jit_compile_list_methods(cu.get()));
}

// [[Rcpp::export]]
SEXP cpp_jit_compile_get_function(SEXP cu, XPtrTorchstring name) {
  auto cu_ = Rcpp::as<XPtrTorchCompilationUnit>(cu);
  // this object would be a XPtrTorchFunctionPtr but since it's tied to
  // the compilation unit, we don't need to wrap it with the destructor.
  // also `res` can return a nullptr, thus we check before returning so
  // we don't end up with a null pointer in the R side.
  void* res = lantern_jit_compile_get_method(cu_.get(), name.get());
  if (res) {
    auto ptr = make_xptr<XPtrTorch>(res);
    // we need to protect the compilation unit so it lives at least until
    // the function pointer is dead.
    R_SetExternalPtrProtected(ptr, cu);
    return ptr;
  } else {
    return R_NilValue;
  }
}
