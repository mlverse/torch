#include <torch.h>

#include <algorithm>

// [[Rcpp::export]]
Rcpp::XPtr<std::nullptr_t> cpp_nullptr() {
  return make_xptr<std::nullptr_t>(nullptr);
}

// [[Rcpp::export]]
Rcpp::XPtr<std::nullptr_t> cpp_nullopt() {
  return make_xptr<std::nullptr_t>(nullptr);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_tensor_undefined() {
  return XPtrTorchTensor(lantern_Tensor_undefined());
}

// [[Rcpp::export]]
XPtrTorchTensor to_index_tensor(XPtrTorchTensor t) {
  // check that there's no zeros
  bool zeros = lantern_Tensor_has_any_zeros(t.get());
  if (zeros) {
    Rcpp::stop("Indexing starts at 1 but found a 0.");
  }

  /// make it 0 based!
  XPtrTorchTensor sign = lantern_Tensor_signbit_tensor(t.get());
  sign = lantern_logical_not_tensor(sign.get());

  // cast from bool to int
  XPtrTorchTensorOptions options = lantern_TensorOptions();
  options = lantern_TensorOptions_dtype(
      options.get(), XPtrTorchDtype(lantern_Dtype_int64()).get());
  sign = lantern_Tensor_to(sign.get(), options.get());

  // create a 1 scalar
  int al = 1;
  XPtrTorchScalar alpha =
      lantern_Scalar((void*)&al, std::string("int").c_str());
  XPtrTorchTensor zero_index =
      lantern_Tensor_sub_tensor_tensor_scalar(t.get(), sign.get(), alpha.get());

  return zero_index;
}

// [[Rcpp::export]]
bool cpp_torch_namespace__use_cudnn_rnn_flatten_weight() {
  return lantern__use_cudnn_rnn_flatten_weight();
}

// [[Rcpp::export]]
void cpp_torch_namespace__store_main_thread_id() {
  // called upon package load to remember the thread ID of the main thread
  main_thread_id();
}

std::thread::id main_thread_id() noexcept {
  static const auto tid = std::this_thread::get_id();

  return tid;
}

// [[Rcpp::export]]
Rcpp::List transpose2(Rcpp::List x) {
  auto templ = Rcpp::as<Rcpp::List>(x[0]);
  auto num_elements = templ.length();

  auto size = x.length();
  std::vector<Rcpp::List> out;

  for (auto i = 0; i < num_elements; i++) {
    out.push_back(Rcpp::List(size));
  }

  for (size_t j = 0; j < size; j++) {
    auto el = Rcpp::as<Rcpp::List>(x[j]);
    for (auto i = 0; i < num_elements; i++) {
      out[i][j] = el[i];
    }
  }

  Rcpp::List ret;
  for (auto i = 0; i < num_elements; i++) {
    ret.push_back(out[i]);
  }

  ret.names() = templ.names();

  return ret;
}

SEXP r7_env = Rf_install("r7_env");
SEXP r7_pvt = Rf_install("r7_pvt");

SEXP r7_self = Rf_install("self");
SEXP r7_private = Rf_install("private");

SEXP r7_generators = Rcpp::Environment::namespace_env("torch").find(".generators");
SEXP r7_enclosing_env = Rcpp::Environment::namespace_env("torch").find(".r7_env");
Rcpp::Function new_env("new.env");

SEXP r7_pvt_class () {
  return Rcpp::CharacterVector({"R7", "R7_private"});
}

SEXP r7_class_env (SEXP self) {
  return Rf_findVarInFrame(r7_generators, Rf_installChar(STRING_ELT(Rf_getAttrib(self, R_ClassSymbol), 0)));
}

SEXP r7_get_object_from_env (SEXP env, SEXP name) {
  SEXP object = Rf_findVar(Rf_installChar(STRING_ELT(name, 0)), env);
  if (object == R_UnboundValue) {
    object = R_NilValue;
  }
  return object;
}

SEXP r7_enclos_env (SEXP self, SEXP pvt) {
#if defined(R_VERSION) && R_VERSION >= R_Version(4, 1, 0)
  SEXP env = PROTECT(R_NewEnv(r7_enclosing_env, 1, 2));
#else
  SEXP env = new_env(1, r7_enclosing_env, 2);
#endif
  Rf_defineVar(r7_self, self, env);
  Rf_defineVar(r7_private, pvt, env);
  UNPROTECT(1);
  return env;
}

SEXP get_private (SEXP self) {
  SEXP e = r7_class_env(self);
  SEXP obj = Rf_findVar(r7_private, e);
  
  SEXP pvt = PROTECT(Rf_allocVector(VECSXP, 0));
  Rf_setAttrib(pvt, r7_env, r7_enclos_env(self, R_NilValue));
  Rf_setAttrib(pvt, r7_pvt, obj);
  Rf_setAttrib(pvt, R_ClassSymbol, r7_pvt_class());
  UNPROTECT(1);
  return pvt;
}

// [[Rcpp::export]]
SEXP extract_method_c (SEXP self, SEXP name) {
  
  if (Rcpp::as<std::string>(name) == "private") {
    return get_private(self);
  }
  
  bool is_private = Rf_inherits(self, "R7_private");
  SEXP e;
  if (is_private) {
    e = Rf_getAttrib(self, r7_pvt);
  } else {
    e = r7_class_env(self);
  }
  
  SEXP obj = r7_get_object_from_env(e, name);
  
  if (!Rf_isFunction(obj)) {
    return obj;
  }
  
  obj = PROTECT(Rf_duplicate(obj));
  if (is_private) {
    SET_CLOENV(obj, Rf_getAttrib(self, r7_env));
  } else {
    SET_CLOENV(obj, r7_enclos_env(self, get_private(self)));
  }
  
  UNPROTECT(1);
  return obj;
}


