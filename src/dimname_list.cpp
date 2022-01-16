
#include <torch.h>

// [[Rcpp::export]]
XPtrTorchDimname cpp_torch_dimname(XPtrTorchstring str) {
  XPtrTorchDimname out = lantern_Dimname(str.get());
  return XPtrTorchDimname(out);
}

// [[Rcpp::export]]
XPtrTorchDimnameList cpp_torch_dimname_list(const Rcpp::List& x) {
  XPtrTorchDimnameList out = lantern_DimnameList();

  for (int i = 0; i < x.length(); i++) {
    lantern_DimnameList_push_back(out.get(),
                                  Rcpp::as<Rcpp::XPtr<XPtrTorch>>(x[i])->get());
  }

  return XPtrTorchDimnameList(out);
}

// [[Rcpp::export]]
std::string cpp_dimname_to_string(XPtrTorchDimname x) {
  return lantern_Dimname_to_string(x.get());
};

// [[Rcpp::export]]
std::vector<std::string> cpp_dimname_list_to_string(XPtrTorchDimnameList x) {
  int64_t size = lantern_DimnameList_size(x.get());
  std::vector<std::string> result;

  for (int i = 0; i < size; i++) {
    auto dimname = XPtrTorchDimname(lantern_DimnameList_at(x.get(), i));
    auto v = lantern_Dimname_to_string(dimname.get());
    result.push_back(std::string(v));
    lantern_const_char_delete(v);
  }

  return result;
};
