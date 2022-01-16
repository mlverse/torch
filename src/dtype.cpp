#include <torch.h>

// [[Rcpp::export]]
std::string cpp_dtype_to_string(XPtrTorchDtype dtype) {
  auto s = lantern_Dtype_type(dtype.get());
  auto out = std::string(s);
  lantern_const_char_delete(s);
  return out;
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchDtype> cpp_torch_float32() {
  return make_xptr<XPtrTorchDtype>(lantern_Dtype_float32());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchDtype> cpp_torch_float64() {
  return make_xptr<XPtrTorchDtype>(lantern_Dtype_float64());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchDtype> cpp_torch_float16() {
  return make_xptr<XPtrTorchDtype>(lantern_Dtype_float16());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchDtype> cpp_torch_uint8() {
  return make_xptr<XPtrTorchDtype>(lantern_Dtype_uint8());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchDtype> cpp_torch_int8() {
  return make_xptr<XPtrTorchDtype>(lantern_Dtype_int8());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchDtype> cpp_torch_int16() {
  return make_xptr<XPtrTorchDtype>(lantern_Dtype_int16());
}

// [[Rcpp::export]]
XPtrTorchDtype cpp_torch_int32() {
  return XPtrTorchDtype(lantern_Dtype_int32());
}

// [[Rcpp::export]]
XPtrTorchDtype cpp_torch_int64() {
  return XPtrTorchDtype(lantern_Dtype_int64());
}

// [[Rcpp::export]]
XPtrTorchDtype cpp_torch_bool() { return XPtrTorchDtype(lantern_Dtype_bool()); }

// [[Rcpp::export]]
XPtrTorchDtype cpp_torch_quint8() {
  return XPtrTorchDtype(lantern_Dtype_quint8());
}

// [[Rcpp::export]]
XPtrTorchDtype cpp_torch_qint8() {
  return XPtrTorchDtype(lantern_Dtype_qint8());
}

// [[Rcpp::export]]
XPtrTorchDtype cpp_torch_qint32() {
  return XPtrTorchDtype(lantern_Dtype_qint32());
}

// [[Rcpp::export]]
void cpp_set_default_dtype(XPtrTorchDtype x) {
  lantern_set_default_dtype(x.get());
}

// [[Rcpp::export]]
XPtrTorchDtype cpp_get_default_dtype() {
  XPtrTorchDtype out = lantern_get_default_dtype();
  return XPtrTorchDtype(out);
}
