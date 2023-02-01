#include <iostream>
#define LANTERN_BUILD
#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"

void _lantern_set_num_threads(int n) {
  LANTERN_FUNCTION_START
  at::set_num_threads(n);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_set_num_interop_threads(int n) {
  LANTERN_FUNCTION_START
  at::set_num_interop_threads(n);
  LANTERN_FUNCTION_END_VOID
}

int _lantern_get_num_threads() {
  LANTERN_FUNCTION_START
  return at::get_num_threads();
  LANTERN_FUNCTION_END_RET(0)
}

int _lantern_get_num_interop_threads() {
  LANTERN_FUNCTION_START
  return at::get_num_interop_threads();
  LANTERN_FUNCTION_END_RET(0)
}