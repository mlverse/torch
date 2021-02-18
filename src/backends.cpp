#include "torch_types.h"
#include "utils.h"

// [[Rcpp::export]]
bool cpp_backends_mkldnn_is_available ()
{
  return lantern_backend_has_mkldnn();
}

// [[Rcpp::export]]
bool cpp_backends_mkl_is_available ()
{
  return lantern_backend_has_mkl();
}

// [[Rcpp::export]]
bool cpp_backends_openmp_is_available ()
{
  return lantern_backend_has_openmp();
}