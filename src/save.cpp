#include <torch.h>

// [[Rcpp::export]]
SEXP cpp_tensor_save(Rcpp::XPtr<XPtrTorchTensor> x, bool base64) {
  if (base64) {
    return torch::string(lantern_tensor_save(x->get(), base64));  
  } else {
    torch::string out = lantern_tensor_save(x->get(), base64);
    
    const char* v = lantern_string_get(out.get());
    auto output = std::string(v, lantern_string_size(out.get()));
    lantern_const_char_delete(v);
    
    Rcpp::RawVector raw_vec(output.size());
    memcpy(&raw_vec[0], output.c_str(), output.size());
    
    return raw_vec;
  }
} 

// [[Rcpp::export]]
XPtrTorchTensor cpp_tensor_load(SEXP input, XPtrTorchOptionalDevice device, bool base64) {
  if (base64) {
    torch::string v = Rcpp::as<torch::string>(input);
    return torch::Tensor(lantern_tensor_load(v.get(), device.get(), base64));
  } else {
    auto raw_vec = Rcpp::as<Rcpp::RawVector>(input);
    torch::string v = std::string((char*)&raw_vec[0], raw_vec.size());
    return torch::Tensor(lantern_tensor_load(v.get(), device.get(), base64));
  }
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
