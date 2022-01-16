#include <torch.h>

// [[Rcpp::export]]
XPtrTorchScalar cpp_torch_scalar(SEXP x) {
  XPtrTorchScalar out;
  std::string type;

  int i;
  double d;
  bool b;

  switch (TYPEOF(x)) {
    case INTSXP:
      i = Rcpp::as<int>(x);
      type = "int";
      out = lantern_Scalar((void*)(&i), type.c_str());
      break;
    case REALSXP:
      d = Rcpp::as<double>(x);
      type = "double";
      out = lantern_Scalar((void*)(&d), type.c_str());
      break;
    case LGLSXP:
      b = Rcpp::as<bool>(x);
      type = "bool";
      out = lantern_Scalar((void*)(&b), type.c_str());
      break;
    case CHARSXP:
      Rcpp::stop("strings are not handled yet");
    default:
      Rcpp::stop("not handled");
  }

  return XPtrTorchScalar(out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchScalarType> cpp_torch_scalar_dtype(
    Rcpp::XPtr<XPtrTorchScalar> self) {
  XPtrTorchScalarType out = lantern_Scalar_dtype(self->get());
  return make_xptr<XPtrTorchScalarType>(out);
}

// [[Rcpp::export]]
int cpp_torch_scalar_to_int(Rcpp::XPtr<XPtrTorchScalar> self) {
  return lantern_Scalar_to_int(self->get());
}

// [[Rcpp::export]]
double cpp_torch_scalar_to_double(Rcpp::XPtr<XPtrTorchScalar> self) {
  return lantern_Scalar_to_double(self->get());
}

// [[Rcpp::export]]
float cpp_torch_scalar_to_float(Rcpp::XPtr<XPtrTorchScalar> self) {
  return lantern_Scalar_to_float(self->get());
}

// [[Rcpp::export]]
bool cpp_torch_scalar_to_bool(Rcpp::XPtr<XPtrTorchScalar> self) {
  return lantern_Scalar_to_bool(self->get());
}
