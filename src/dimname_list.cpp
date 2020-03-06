
#include "torchr_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_torch_dimname(const std::string& str) {
  auto out = lantern_Dimname(str.c_str());
  return make_xptr<XPtrTorch>(out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_torch_dimname_list (const Rcpp::List& x) {
  auto out = lantern_DimnameList();

  for (int i = 0; i < x.length(); i++) {
    lantern_DimnameList_push_back(out, Rcpp::as<Rcpp::XPtr<XPtrTorch>>(x[i])->get());
  }

  return make_xptr<XPtrTorch>(out);
}
// 
// // [[Rcpp::export]]
// std::string cpp_dimname_to_string (Rcpp::XPtr<torch::Dimname> x) {
//   torch::Dimname out = *x;
//   return out.symbol().toUnqualString();
// };
// 
// // [[Rcpp::export]]
// std::vector<std::string> cpp_dimname_list_to_string (Rcpp::XPtr<std::vector<torch::Dimname>> x) {
//   auto out = *x;
//   std::vector<std::string> result;
//   
//   for (int i = 0; i < out.size(); i++) {
//     result.push_back(out[i].symbol().toUnqualString());
//   }
//   
//   return result;
// };
//  