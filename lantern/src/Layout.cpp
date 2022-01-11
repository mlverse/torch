#include <iostream>

#define LANTERN_BUILD

#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"
void *_lantern_Layout_strided() {
  LANTERN_FUNCTION_START
  return make_raw::Layout(torch::kStrided);
  LANTERN_FUNCTION_END
}

void *_lantern_Layout_sparse() {
  LANTERN_FUNCTION_START
  return make_raw::Layout(torch::kSparse);
  LANTERN_FUNCTION_END
}

const char *_lantern_Layout_string(void *x) {
  LANTERN_FUNCTION_START
  std::string str;
  auto l = from_raw::Layout(x);
  if (l == torch::kStrided) {
    str = "strided";
  } else if (l == torch::kSparse) {
    str = "sparse_coo";
  }

  char *cstr = new char[str.length() + 1];
  strcpy(cstr, str.c_str());
  return cstr;
  LANTERN_FUNCTION_END
}