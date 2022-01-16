#include <torch.h>

// [[Rcpp::export]]
std::string cpp_tensor_save(Rcpp::XPtr<XPtrTorchTensor> x) {
  const char* s = lantern_tensor_save(x->get());
  auto out = std::string(s);
  lantern_const_char_delete(s);
  return out;
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_tensor_load(std::string s, XPtrTorchOptionalDevice device) {
  XPtrTorchTensor t = lantern_tensor_load(s.c_str(), device.get());
  return t;
}

// [[Rcpp::export]]
Rcpp::List cpp_load_state_dict(std::string path) {
  XPtrTorchIValue v = lantern_load_state_dict(path.c_str());

  XPtrTorchTensorList values = lantern_get_state_dict_values(v.get());

  XPtrTorchvector_string s = lantern_get_state_dict_keys(v.get());
  int size = lantern_vector_string_size(s.get());

  std::vector<std::string> keys;
  for (int i = 0; i < size; i++) {
    const char* k = lantern_vector_string_at(s.get(), i);
    keys.push_back(std::string(k));
    lantern_const_char_delete(k);
  }

  Rcpp::List L =
      Rcpp::List::create(Rcpp::Named("keys") = keys,
                         Rcpp::Named("values") = XPtrTorchTensorList(values));

  return L;
}
