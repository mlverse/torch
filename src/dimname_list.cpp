
#include "torchr_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchDimname> cpp_torch_dimname(const std::string& str) {
  XPtrTorchDimname out = lantern_Dimname(str.c_str());
  return make_xptr<XPtrTorchDimname>(out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchDimnameList> cpp_torch_dimname_list (const Rcpp::List& x) {
  XPtrTorchDimnameList out = lantern_DimnameList();

  for (int i = 0; i < x.length(); i++) {
    lantern_DimnameList_push_back(out.get(), Rcpp::as<Rcpp::XPtr<XPtrTorch>>(x[i])->get());
  }

  return make_xptr<XPtrTorchDimnameList>(out);
}

// [[Rcpp::export]]
std::string cpp_dimname_to_string (Rcpp::XPtr<XPtrTorchDimname> x) {
  return lantern_Dimname_to_string(x->get());
};


// [[Rcpp::export]]
std::vector<std::string> cpp_dimname_list_to_string (Rcpp::XPtr<XPtrTorchDimnameList> x) {
  
  int64_t size = lantern_DimnameList_size(x->get());
  std::vector<std::string> result;
  
  for (int i = 0; i < size; i++) {
    result.push_back(lantern_Dimname_to_string(XPtrTorchDimname(lantern_DimnameList_at(x->get(), i)).get()));
  }

  return result;
};
