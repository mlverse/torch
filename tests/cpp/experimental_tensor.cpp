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

void compile_experimental_creation_api(const ExperimentalTensor& input,
                                       torch::TensorOptions options) {
  using namespace torch::experimental;
  auto a = empty({2, 3});
  auto b = empty({2, 3}, options);
  auto c = empty_strided({2, 3}, {3, 1});
  auto d = zeros({2, 3});
  auto e = ones({2, 3}, options);
  auto f = rand({2, 3});
  auto g = randn({2, 3}, options);
  auto h = randint(10, std::vector<std::int64_t>{2, 3});
  auto i = randint(2, 10, std::vector<std::int64_t>{2, 3}, options);
  auto j = randperm(10);
  auto k = arange(10);
  auto l = arange(1, 10, 2);
  auto m = linspace(1, 10, 5);
  auto n = logspace(1, 3, 3, 10);
  auto o = eye(3);
  auto p = eye(3, 2, options);
  auto q = full({2, 3}, 4);
  auto r = scalar_tensor(2);
  auto s = empty_like(input);
  auto t = zeros_like(input);
  auto u = ones_like(input, options);
  auto v = rand_like(input);
  auto w = randn_like(input, options);
  auto x = randint_like(input, 2, 10);
  auto y = full_like(input, 3, options);
  (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g;
  (void)h; (void)i; (void)j; (void)k; (void)l; (void)m; (void)n;
  (void)o; (void)p; (void)q; (void)r; (void)s; (void)t; (void)u;
  (void)v; (void)w; (void)x; (void)y;
}

// [[Rcpp::export]]
torch::experimental::Tensor cpp_experimental_creation_zeros() {
  return torch::experimental::zeros({2, 3});
}

// [[Rcpp::export]]
torch::experimental::Tensor cpp_experimental_creation_arange() {
  return torch::experimental::arange(1, 8, 2);
}

// [[Rcpp::export]]
torch::experimental::Tensor cpp_experimental_creation_eye() {
  return torch::experimental::eye(3, 2);
}

// [[Rcpp::export]]
torch::experimental::Tensor cpp_experimental_creation_full_like(
    torch::experimental::Tensor input) {
  return torch::experimental::full_like(input, 7);
}

// [[Rcpp::export]]
std::vector<std::int64_t> cpp_experimental_creation_random_sizes() {
  return torch::experimental::randint(0, 10,
      std::vector<std::int64_t>{2, 3}).sizes();
}
