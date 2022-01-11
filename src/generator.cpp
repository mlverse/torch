#include <torch.h>

// https://pytorch.org/docs/stable/torch.html#generators
// https://github.com/pytorch/pytorch/blob/f531815526c69f432e46fadece44f5d3a9b70e30/torch/csrc/Generator.cpp

// [[Rcpp::export]]
XPtrTorchGenerator cpp_torch_generator() {
  XPtrTorchGenerator out = lantern_Generator();
  return XPtrTorchGenerator(out);
}

// [[Rcpp::export]]
std::string cpp_generator_current_seed(XPtrTorchGenerator generator) {
  uint64_t seed = lantern_Generator_current_seed(generator.get());
  auto seed_str = std::to_string(seed);
  return seed_str;
}

// [[Rcpp::export]]
void cpp_generator_set_current_seed(XPtrTorchGenerator generator,
                                    std::string seed) {
  uint64_t value;
  std::istringstream iss(seed);
  iss >> value;

  lantern_Generator_set_current_seed(generator.get(), value);
}

// [[Rcpp::export]]
void cpp_torch_manual_seed(std::string seed) {
  int64_t value;
  std::istringstream iss(seed);
  iss >> value;

  lantern_manual_seed(value);
}
