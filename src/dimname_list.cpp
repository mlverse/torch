#include "torch_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
Rcpp::XPtr<torch::Dimname> cpp_torch_dimname(const std::string& str) {
  return make_xptr<torch::Dimname>(torch::Dimname::fromSymbol(torch::Symbol::dimname(str)));
}


// [[Rcpp::export]]
Rcpp::XPtr<std::vector<torch::Dimname>> cpp_torch_dimname_list (const Rcpp::List& x) {
  
  std::vector<torch::Dimname> out;
  
  for (int i = 0; i < x.length(); i++) {
    out.push_back(*Rcpp::as<Rcpp::XPtr<torch::Dimname>>(x.at(i)));
  }
  
  return make_xptr<std::vector<torch::Dimname>>(out);
}

// [[Rcpp::export]]
std::string cpp_dimname_to_string (Rcpp::XPtr<torch::Dimname> x) {
  torch::Dimname out = *x;
  return out.symbol().toUnqualString();
};

// [[Rcpp::export]]
std::vector<std::string> cpp_dimname_list_to_string (Rcpp::XPtr<std::vector<torch::Dimname>> x) {
  auto out = *x;
  std::vector<std::string> result;
  
  for (int i = 0; i < out.size(); i++) {
    result.push_back(out[i].symbol().toUnqualString());
  }
  
  return result;
};