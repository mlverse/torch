
#include <torch/torch.h>
#include "utils.hpp"

void* lanternFromBlob (void* data, void* sizes, size_t sizes_size) {
  return (void *) new LanternObject<torch::Tensor>(torch::from_blob(
      data, 
      std::vector<int64_t>(reinterpret_cast<int64_t *>(sizes), reinterpret_cast<int64_t *>(sizes) + sizes_size),
      torch::kDouble
  ));
}

char* lanternTensorStreamInsertion (void* x) {
  std::stringstream ss;
  ss << *reinterpret_cast<torch::Tensor *>(x);
  auto str = ss.str();
  char *cstr = new char[str.length() + 1];
  strcpy(cstr, str.c_str());
  return cstr;
}


