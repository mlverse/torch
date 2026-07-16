#include <Rcpp.h>

// [[Rcpp::depends(torch)]]
#define IMPORT_TORCH
#include <torch/experimental.h>

#include <type_traits>

using ExperimentalTensor = torch::experimental::Tensor;

static_assert(std::is_copy_constructible<ExperimentalTensor>::value, "Tensor must be copyable");
static_assert(std::is_move_constructible<ExperimentalTensor>::value, "Tensor must be movable");
static_assert(std::is_same<decltype(std::declval<const ExperimentalTensor&>().sizes()),
                           std::vector<std::int64_t>>::value,
              "sizes() must return a standard owning vector");

void compile_experimental_tensor_api(ExperimentalTensor x,
                                     const ExperimentalTensor& y,
                                     torch::TensorOptions options) {
  auto clone = x.clone();
  auto reshaped = x.reshape({-1});
  auto transposed = x.transpose(0, 1);
  auto product = x.matmul(y);
  auto arithmetic = (x + y) * y;
  auto converted = x.to(options);
  x.requires_grad_().relu_().zero_();

  (void)clone;
  (void)reshaped;
  (void)transposed;
  (void)product;
  (void)arithmetic;
  (void)converted;
}

// [[Rcpp::export]]
torch::experimental::Tensor cpp_experimental_tensor_relu_reshape(
    torch::experimental::Tensor input) {
  return input.relu().reshape({-1});
}

// [[Rcpp::export]]
std::vector<std::int64_t> cpp_experimental_tensor_sizes(SEXP input) {
  return torch::experimental::from_torch(from_sexp_tensor(input)).sizes();
}

// [[Rcpp::export]]
torch::experimental::Tensor cpp_experimental_tensor_generated_methods(
    torch::experimental::Tensor input) {
  return input.abs().square().sin();
}

// [[Rcpp::export]]
torch::experimental::Tensor cpp_experimental_tensor_native_integer_argument(
    torch::experimental::Tensor input) {
  return input.select(0, 0);
}

// [[Rcpp::export]]
bool cpp_experimental_tensor_native_bool_return(
    torch::experimental::Tensor input) {
  return input.equal(input);
}
