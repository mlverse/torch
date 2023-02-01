#include <iostream>

#define LANTERN_BUILD

#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"

void *_lantern_Tensor_storage(void *self) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  return make_raw::Storage(x.storage());
  LANTERN_FUNCTION_END
}

bool _lantern_Tensor_has_storage(void *self) {
  LANTERN_FUNCTION_START
  torch::Tensor x = from_raw::Tensor(self);
  return x.has_storage();
  LANTERN_FUNCTION_END_RET(false)
}

void *_lantern_Storage_data_ptr(void *self) {
  LANTERN_FUNCTION_START
  auto x = from_raw::Storage(self);
  return x.data_ptr().get();
  LANTERN_FUNCTION_END
}
