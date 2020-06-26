#include "torch_types.h"

using namespace Rcpp;

// [[Rcpp::export]]
void cpp_lantern_init(std::string path) {
  std::string error;
  if (!lanternInit(path, &error))
    Rcpp::stop(error);
}

// [[Rcpp::export]]
void cpp_lantern_test() {
  lanternTest();
}

// [[Rcpp::export]]
bool cpp_lantern_has_error() {
  const char* pLast = lanternLastError();
  return pLast != NULL;
}

// [[Rcpp::export]]
std::string cpp_lantern_last_error() {
  const char* pError = lanternLastError();
  if (pError == NULL)
    return std::string("");
  
  return std::string(pError);
}

// [[Rcpp::export]]
void cpp_lantern_error_clear() {
  lanternLastErrorClear();
}
