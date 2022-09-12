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
