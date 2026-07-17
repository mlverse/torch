#include <Rcpp.h>

#define IMPORT_TORCH
#include <torch/experimental.h>

#include <type_traits>

static_assert(std::is_copy_constructible<torch::experimental::Tensor>::value, "Tensor must be copyable");
static_assert(std::is_move_constructible<torch::experimental::Tensor>::value, "Tensor must be movable");
static_assert(std::is_same<decltype(std::declval<const torch::experimental::Tensor&>().sizes()),
                           std::vector<std::int64_t>>::value,
              "sizes() must return a standard owning vector");

void compile_experimental_tensor_api(torch::experimental::Tensor x,
                                     const torch::experimental::Tensor& y,
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

// Exercise native scalar overload resolution independently of the nn2poly
// compatibility examples below.
// [[Rcpp::export]]
torch::experimental::Tensor cpp_experimental_tensor_scalar_overloads(
    torch::experimental::Tensor input) {
  return input.add(2).mul(3.0).pow(2);
}

// [[Rcpp::export]]
torch::experimental::Tensor cpp_experimental_tensor_inplace_overloads(
    torch::experimental::Tensor input, torch::experimental::Tensor other) {
  auto output = input.clone();
  output.add_(other);
  output.select(0, 0).fill_(7);
  return output;
}

// [[Rcpp::export]]
torch::experimental::Tensor cpp_experimental_tensor_cat_overloads(
    torch::experimental::Tensor first, torch::experimental::Tensor second) {
  return torch::experimental::cat({first, second}, 0);
}

// [[Rcpp::export]]
torch::experimental::Tensor cpp_public_torch_namespace_functions(
    torch::experimental::Tensor first, torch::experimental::Tensor second) {
  auto joined = torch::cat({first, second}, 0);
  auto bias = torch::ones({joined.size(0)});
  return torch::matmul(joined.reshape({1, -1}), bias.reshape({-1, 1}));
}

// [[Rcpp::export]]
torch::experimental::Tensor cpp_public_torch_nested_namespace_functions(
    torch::experimental::Tensor input) {
  auto diagonal = torch::linalg::diagonal(torch::abs(input), 0, 0, 1);
  return torch::special::expm1(diagonal);
}

// [[Rcpp::export]]
torch::experimental::Tensor cpp_experimental_tensor_to_device(torch::experimental::Tensor input) {
  return input.to(torch::kCPU);
}

void compile_experimental_tensor_to_indexed_device(
    const torch::experimental::Tensor& input) {
  auto cpu = input.to(torch::kCPU, 0);
  auto cuda = input.to(torch::kCUDA, 0);
  (void)cpu;
  (void)cuda;
}

void compile_public_torch_fft_namespace(
    const torch::experimental::Tensor& input, XPtrTorchoptional_int64_t n,
    XPtrTorchoptional_string_view norm) {
  auto transformed = torch::fft::fft(input, n, -1, norm);
  (void)transformed;
}

void compile_experimental_creation_api(const torch::experimental::Tensor& input,
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

void compile_public_torch_namespace_api(const torch::experimental::Tensor& input) {
  auto a = torch::empty({2, 3});
  auto b = torch::empty_strided({2, 3}, {3, 1});
  auto c = torch::zeros({2, 3});
  auto d = torch::ones({2, 3});
  auto e = torch::rand({2, 3});
  auto f = torch::randn({2, 3});
  auto g = torch::randint(10, std::vector<std::int64_t>{2, 3});
  auto h = torch::randperm(10);
  auto i = torch::arange(10);
  auto j = torch::linspace(0, 1, 5);
  auto k = torch::logspace(0, 2, 3);
  auto l = torch::eye(3);
  auto m = torch::full({2, 3}, 4);
  auto n = torch::scalar_tensor(2);
  auto o = torch::empty_like(input);
  auto p = torch::zeros_like(input);
  auto q = torch::ones_like(input);
  auto r = torch::rand_like(input);
  auto s = torch::randn_like(input);
  auto t = torch::randint_like(input, 0, 10);
  auto u = torch::full_like(input, 3);
  auto v = torch::cat({input, input});
  auto w = torch::matmul(input, input);
  (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g;
  (void)h; (void)i; (void)j; (void)k; (void)l; (void)m; (void)n;
  (void)o; (void)p; (void)q; (void)r; (void)s; (void)t; (void)u;
  (void)v; (void)w;
}

void compile_experimental_creation_types() {
  using namespace torch::experimental;
  ScalarType scalar_types[] = {
      kFloat32, kFloat64, kFloat16, kBFloat16, kComplexHalf, kComplexFloat,
      kComplexDouble, kFloat8E4M3FN, kFloat8E5M2, kUInt8, kInt8, kInt16,
      kInt32, kInt64, kBool, kQUInt8, kQInt8, kQInt32, kByte, kChar, kShort,
      kInt, kLong, kHalf, kFloat, kDouble, kComplexFloat32, kComplexFloat64,
      kComplexFloat128};
  Layout layouts[] = {
      kStrided, kSparse, kSparseCsr, kSparseCsc, kSparseBsr, kSparseBsc};
  DeviceType devices[] = {kCPU, kCUDA};
  auto options = TensorOptions()
                     .dtype(kFloat64)
                     .layout(kStrided)
                     .device(kCPU)
                     .device(kCUDA, 0)
                     .requires_grad()
                     .pinned_memory(false);
  (void)scalar_types;
  (void)layouts;
  (void)devices;
  (void)options;
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

// This is also a complete sourceCpp-style usage example.
// [[Rcpp::export]]
torch::experimental::Tensor zeros(int rows, int cols) {
  auto options = torch::experimental::TensorOptions()
                     .dtype(torch::experimental::kFloat64);
  return torch::experimental::zeros({rows, cols}, options);
}

// Torch equivalents of every operation used by nn2poly's linalg_arma.h.
// [[Rcpp::export]]
torch::experimental::Tensor cpp_experimental_nn2poly_linear(
    torch::experimental::Tensor layer, torch::experimental::Tensor coefficients) {
  using namespace torch::experimental;
  auto intercept = zeros({1, coefficients.size(1)});
  intercept.select(1, 0).fill_(1.0);
  return layer.t().matmul(cat({intercept, coefficients}, 0));
}

// [[Rcpp::export]]
torch::experimental::Tensor cpp_experimental_nn2poly_add_partition(
    torch::experimental::Tensor matrix, std::int64_t column, double scalar,
    torch::experimental::Tensor values) {
  auto result = matrix.clone();
  result.select(1, column).add_(values.mul(scalar));
  return result;
}

// [[Rcpp::export]]
torch::experimental::Tensor cpp_experimental_nn2poly_add_poly_eval(
    torch::experimental::Tensor matrix, std::int64_t column, torch::experimental::Tensor input,
    std::int64_t input_column, std::vector<double> coefficients) {
  using namespace torch::experimental;
  auto result = full({input.size(0)}, coefficients.back());
  auto x = input.select(1, input_column);
  for (std::int64_t i = static_cast<std::int64_t>(coefficients.size()) - 2;
       i >= 0; --i) {
    result = result.mul(x).add(coefficients[static_cast<std::size_t>(i)]);
  }
  auto output = matrix.clone();
  output.select(1, column).add_(result);
  return output;
}

// [[Rcpp::export]]
torch::experimental::Tensor cpp_experimental_nn2poly_accumulate_partition(
    torch::experimental::Tensor matrix, std::int64_t output_column,
    std::vector<std::int64_t> input_columns,
    std::vector<double> multipliers, torch::experimental::Tensor output) {
  using namespace torch::experimental;
  auto partition = ones({matrix.size(0)});
  for (std::size_t i = 0; i < input_columns.size(); ++i) {
    partition = partition.mul(
        matrix.select(1, input_columns[i]).pow(multipliers[i]));
  }
  auto result = output.clone();
  result.select(1, output_column)
      .add_(partition.mul(matrix.select(1, 0).pow(multipliers.front())));
  return result;
}

// Keep the fixture in one translation unit because torch_imports.h provides
// package-level C-callable shims. RcppExports.cpp is generated normally by
// Rcpp::compileAttributes(), then compiled here rather than as a second object.
#include "RcppExports.cpp"
