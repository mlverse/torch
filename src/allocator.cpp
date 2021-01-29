#include "torch_types.h"
#include "utils.h"

void call_r_gc ()
{
  static Rcpp::Environment torch("package:torch");
  static Rcpp::Function gc = torch["torch_gc_wrapper"];
  gc();
}

// [[Rcpp::export]]
void cpp_set_lantern_allocator (uint64_t threshold_call_gc = 4000)
{
  set_lantern_allocator(&call_r_gc, threshold_call_gc);
}