#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void* lantern_from_blob (void* data, void* sizes, size_t sizes_size, void* options) {
  return (void *) new LanternObject<torch::Tensor>(torch::from_blob(
      data, 
      std::vector<int64_t>(reinterpret_cast<int64_t *>(sizes), reinterpret_cast<int64_t *>(sizes) + sizes_size),
      *reinterpret_cast<torch::TensorOptions *>(options)
  ));
}

char* lantern_Tensor_StreamInsertion (void* x) {
  std::stringstream ss;
  ss << *reinterpret_cast<torch::Tensor *>(x);
  auto str = ss.str();
  char *cstr = new char[str.length() + 1];
  strcpy(cstr, str.c_str());
  return cstr;
}

