#include "torch_types.h"
#include "utils.hpp"

// equivalent to (n-1):0 in R
std::vector<int64_t> revert_int_seq (int n) {
  std::vector<int64_t> l(n);
  std::iota(l.begin(), l.end(), 0);
  std::reverse(l.begin(), l.end());
  return l;
};

template <int RTYPE, torch::Dtype DTYPE>
torch::Tensor tensor_from_r_array (const SEXP x, const std::vector<int64_t> dim) {

  Rcpp::Vector<RTYPE> vec(x);

  auto options = torch::TensorOptions()
      .dtype(DTYPE)
      .device("cpu");
  
  auto tensor = torch::from_blob(vec.begin(), dim, options);

  if (dim.size() == 1) {
    // if we have a 1-dim vector contigous doesn't trigger a copy, and
    // would be unexpected.
    tensor = tensor.clone();
  }
  
  tensor = tensor
    .permute(revert_int_seq(dim.size()))
    .contiguous();

  return tensor;
};

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> cpp_torch_tensor (SEXP x, std::vector<std::int64_t> dim, 
                                            Rcpp::XPtr<torch::TensorOptions> options, 
                                            bool requires_grad) {

  torch::Tensor tensor;
  
  if (TYPEOF(x) == INTSXP) {
    tensor = tensor_from_r_array<INTSXP, torch::kInt>(x, dim);
  } else if (TYPEOF(x) == REALSXP) {
    tensor = tensor_from_r_array<REALSXP, torch::kDouble>(x, dim);
  } else if (TYPEOF(x) == LGLSXP) {
    tensor = tensor_from_r_array<LGLSXP, torch::kInt32>(x, dim);
  } else {
    Rcpp::stop("R type not handled");
  };
  
  tensor = tensor
    .to(*options)
    .set_requires_grad(requires_grad);
  
  return make_xptr<torch::Tensor>(tensor);
}

template <int RTYPE, typename STDTYPE>
Rcpp::List tensor_to_r_array (torch::Tensor x) {
  
  Rcpp::IntegerVector dimensions(x.ndimension());
  for (int i = 0; i < x.ndimension(); ++i) {
    dimensions[i] = x.size(i);
  }
  
  auto ten = x.contiguous();
  
  Rcpp::Vector<RTYPE> vec(ten.data_ptr<STDTYPE>(), ten.data_ptr<STDTYPE>() + ten.numel());
  
  return Rcpp::List::create(Rcpp::Named("vec") = vec, Rcpp::Named("dim") = dimensions);
}

// [[Rcpp::export]]
Rcpp::List cpp_as_array (Rcpp::XPtr<torch::Tensor> x) {
  
  torch::Tensor ten = *x;
  
  if (ten.dtype() == torch::kInt) {
    return tensor_to_r_array<INTSXP, int32_t>(ten);
  } else if (ten.dtype() == torch::kDouble) {
    return tensor_to_r_array<REALSXP, double>(ten);
  } else if (ten.dtype() == torch::kByte) {
    return tensor_to_r_array<LGLSXP, std::uint8_t>(ten);
  } else if (ten.dtype() == torch::kLong) {
    return tensor_to_r_array<INTSXP, int32_t>(ten.to(torch::kInt));
  } else if (ten.dtype() == torch::kFloat) {
    return tensor_to_r_array<REALSXP, double>(ten.to(torch::kDouble));
  } else if (ten.dtype() == torch::kInt16) {
    return tensor_to_r_array<INTSXP, int16_t>(ten.to(torch::kInt16));
  }
  
  Rcpp::stop("dtype not handled");
};
