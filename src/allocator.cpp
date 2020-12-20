#include "torch_types.h"
#include "utils.h"

void call_r_gc ()
{
  static Rcpp::Function r_gc("gc");
  r_gc(Rcpp::Named("full")=false);
}

// [[Rcpp::export]]
void cpp_set_lantern_allocator ()
{
  set_lantern_allocator(&call_r_gc);
}