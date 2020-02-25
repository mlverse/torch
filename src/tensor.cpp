#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void* lantern_from_blob (void* data, int64_t* sizes, size_t sizes_size, void* options) {
  return (void *) new LanternObject<torch::Tensor>(torch::from_blob(
      data, 
      std::vector<int64_t>(sizes, sizes + sizes_size),
      reinterpret_cast<LanternObject<torch::TensorOptions>*>(options)->get()
  ));
}

const char* lantern_Tensor_StreamInsertion (void* x) {
  std::stringstream ss;
  ss << reinterpret_cast<LanternObject<torch::Tensor> *>(x)->get();
  return ss.str().c_str();
}

