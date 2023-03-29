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

// [[Rcpp::export]]
torch::Tensor cpp_torch_get_rng_state () {
  return torch::Tensor(lantern_cpu_get_rng_state());
}

// [[Rcpp::export]]
void cpp_torch_set_rng_state (torch::Tensor state) {
  lantern_cpu_set_rng_state(state.get());
}

// [[Rcpp::export]]
torch::Tensor cpp_torch_cuda_get_rng_state (int device) {
  return torch::Tensor(lantern_cuda_get_rng_state(device));
}

// [[Rcpp::export]]
void cpp_torch_cuda_set_rng_state (int device, torch::Tensor state) {
  lantern_cuda_set_rng_state(device, state);
}
