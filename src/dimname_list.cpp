#include "torch_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
Rcpp::XPtr<torch::DimnameList> cpp_torch_dimname_list (std::vector<std::string> x) {
  
  std::vector<torch::Dimname> out;
  
  for (int i = 0; i < x.size(); ++i) {
    //Rcpp::Rcout << torch::Dimname::isValidName(x[i]) << std::endl;
    out.push_back(torch::Dimname::fromSymbol(torch::Symbol::dimname(x[i])));
    //Rcpp::Rcout << out[i] << std::endl;
  }
  
  torch::DimnameList result(out);  
  
  return make_xptr<torch::DimnameList>(result);
}

// [[Rcpp::export]]
std::vector<std::string> cpp_torch_dimname_to_string (Rcpp::XPtr<torch::DimnameList> x) {
  
  std::vector<std::string> out;
  torch::DimnameList dnl = *x;
  torch::Dimname a(dnl.at(0));
  Rcpp::Rcout << a << std::endl;
  
  for (int i = 0; i < dnl.size(); i++) {
    out.push_back(dnl.at(i).symbol().toUnqualString());
  }
  
  return out;
};