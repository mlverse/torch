// Included by experimental_tensor.cpp to share its Torch imports.

// [[Rcpp::export]]
Rcpp::List cpp_experimental_creation_integral_scalars() {
  using namespace torch::experimental;
  const std::int64_t value = INT64_C(9007199254740993);
  const auto options = TensorOptions().dtype(kInt64);
  const auto input = zeros({2}, options);
  auto filled = input.clone();
  filled.fill_(value);
  const auto sequence = arange(value, value + 4, std::int64_t(1), options);
  const auto mixed_sequence = arange(value, value + 4, 1, options);

  // Compare int64_t values in C++: converting to R doubles would mask rounding.
  const auto exact = [value](const Tensor& tensor) {
    return tensor.data_ptr<std::int64_t>()[0] == value;
  };
  return Rcpp::List::create(
      Rcpp::Named("scalar") = exact(scalar_tensor(value, options)),
      Rcpp::Named("literal") = exact(scalar_tensor(9007199254740993LL, options)),
      Rcpp::Named("full") = exact(full({2}, value, options)),
      Rcpp::Named("full_vector") = exact(full(std::vector<std::int64_t>{2}, value)),
      Rcpp::Named("full_like") = exact(full_like(input, value)),
      Rcpp::Named("full_like_options") = exact(full_like(input, value, options)),
      Rcpp::Named("fill") = exact(filled),
      Rcpp::Named("arange") = exact(sequence) && sequence.size(0) == 4 &&
          sequence.data_ptr<std::int64_t>()[3] == value + 3,
      Rcpp::Named("mixed_integer_arange") = exact(mixed_sequence),
      Rcpp::Named("integer_full") = SEXP(full({2}, 3)),
      Rcpp::Named("integer_arange") = SEXP(arange(4)),
      Rcpp::Named("integer_arange_options") = SEXP(arange(4, options)),
      Rcpp::Named("integer_arange_start") = SEXP(arange(1, 4)),
      Rcpp::Named("integer_arange_step") = SEXP(arange(1, 5, 2)),
      Rcpp::Named("floating_full") = SEXP(full({2}, 1.5)),
      Rcpp::Named("float_full") = SEXP(full({2}, 1.5f)),
      Rcpp::Named("floating_arange") = SEXP(arange(1.0, 3.0, 0.5)),
      Rcpp::Named("mixed_arange") = SEXP(arange(1, 3, 0.5)),
      Rcpp::Named("scalar_default") = SEXP(scalar_tensor(3)),
      Rcpp::Named("bool_full") = SEXP(full({2}, true)));
}

// [[Rcpp::export]]
void cpp_experimental_creation_unsigned_overflow() {
  torch::full({1}, std::numeric_limits<std::uint64_t>::max());
}
