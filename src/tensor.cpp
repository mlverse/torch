
#include "torchr_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
void cpp_torch_tensor_print (Rcpp::XPtr<XPtrTorch> x) {
  const char* s = lantern_Tensor_StreamInsertion(x->get());
  Rcpp::Rcout << std::string(s) << std::endl;
};

// equivalent to (n-1):0 in R
std::vector<int64_t> revert_int_seq (int n) {
  std::vector<int64_t> l(n);
  std::iota(l.begin(), l.end(), 0);
  std::reverse(l.begin(), l.end());
  return l;
};

template <int RTYPE>
XPtrTorch tensor_from_r_array (const SEXP x, std::vector<int64_t> dim, std::string dtype) {

  Rcpp::Vector<RTYPE> vec(x);

  auto options = lantern_TensorOptions();
  
  if (dtype == "double") {
    options = lantern_TensorOptions_dtype(options, lantern_Dtype_float64());
  } else if (dtype == "int") {
    options = lantern_TensorOptions_dtype(options, lantern_Dtype_int32());
  }

  options = lantern_TensorOptions_device(options, lantern_Device("cpu", 0, false));

  auto tensor = lantern_from_blob(vec.begin(), &dim[0], dim.size(), options);

  if (dim.size() == 1) {
    // if we have a 1-dim vector contigous doesn't trigger a copy, and
    // would be unexpected.
    tensor = lantern_Tensor_clone(tensor);
  }

  auto reverse_dim = revert_int_seq(dim.size());
  tensor = lantern_Tensor_permute(tensor, lantern_vector_int64_t(&reverse_dim[0], reverse_dim.size()));
  tensor = lantern_Tensor_contiguous(tensor);

  return tensor;
};

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorch> cpp_torch_tensor (SEXP x, std::vector<std::int64_t> dim,
                                            Rcpp::XPtr<XPtrTorchTensorOptions> options,
                                            bool requires_grad) {

  XPtrTorch tensor(nullptr);

  if (TYPEOF(x) == INTSXP) {
    tensor = tensor_from_r_array<INTSXP>(x, dim, "int");
  } else if (TYPEOF(x) == REALSXP) {
    tensor = tensor_from_r_array<REALSXP>(x, dim, "double");
  } else if (TYPEOF(x) == LGLSXP) {
    tensor = tensor_from_r_array<LGLSXP>(x, dim, "int");
  } else {
    Rcpp::stop("R type not handled");
  };
  
  tensor = lantern_Tensor_to(tensor.get(), options->get());
  tensor = lantern_Tensor_set_requires_grad(tensor.get(), requires_grad);

  return make_xptr<XPtrTorch>(tensor);
}

Rcpp::IntegerVector tensor_dimensions (XPtrTorch x) {
  int64_t ndim = lantern_Tensor_ndimension(x.get());
  Rcpp::IntegerVector dimensions(ndim);
  for (int i = 0; i < ndim; ++i) {
    dimensions[i] = lantern_Tensor_size(x.get(), i);
  }
  return dimensions;
}

Rcpp::List tensor_to_r_array_double (XPtrTorch x) {
  auto ten = lantern_Tensor_contiguous(x.get());
  auto d_ptr = lantern_Tensor_data_ptr_double(ten);
  Rcpp::Vector<REALSXP> vec(d_ptr, d_ptr + lantern_Tensor_numel(ten));
  return Rcpp::List::create(Rcpp::Named("vec") = vec, Rcpp::Named("dim") = tensor_dimensions(x));
}

Rcpp::List tensor_to_r_array_uint8_t (XPtrTorch x) {
  auto ten = lantern_Tensor_contiguous(x.get());
  auto d_ptr = lantern_Tensor_data_ptr_uint8_t(ten);
  Rcpp::Vector<LGLSXP> vec(d_ptr, d_ptr + lantern_Tensor_numel(ten));
  return Rcpp::List::create(Rcpp::Named("vec") = vec, Rcpp::Named("dim") = tensor_dimensions(x));
}

Rcpp::List tensor_to_r_array_int32_t (XPtrTorch x) {
  auto ten = lantern_Tensor_contiguous(x.get());
  auto d_ptr = lantern_Tensor_data_ptr_int32_t(ten);
  Rcpp::Vector<INTSXP> vec(d_ptr, d_ptr + lantern_Tensor_numel(ten));
  return Rcpp::List::create(Rcpp::Named("vec") = vec, Rcpp::Named("dim") = tensor_dimensions(x));
}

Rcpp::List tensor_to_r_array_bool (XPtrTorch x) {
  auto ten = lantern_Tensor_contiguous(x.get());
  auto d_ptr = lantern_Tensor_data_ptr_bool(ten);
  Rcpp::Vector<LGLSXP> vec(d_ptr, d_ptr + lantern_Tensor_numel(ten));
  return Rcpp::List::create(Rcpp::Named("vec") = vec, Rcpp::Named("dim") = tensor_dimensions(x));
}

// [[Rcpp::export]]
Rcpp::List cpp_as_array (Rcpp::XPtr<XPtrTorch> x) {

  auto ten = x->get();
  std::string dtype = lantern_Dtype_type(lantern_Tensor_dtype(ten));
  
   
  if (dtype == "Byte") {
    return tensor_to_r_array_uint8_t(ten);
  } 
  
  if (dtype == "Int") {
    return tensor_to_r_array_int32_t(ten);
  }
  
  if (dtype == "Bool") {
    return tensor_to_r_array_bool(ten);
  }
  
  if (dtype == "Double") {
    return tensor_to_r_array_double(ten);
  }
  
  auto options = lantern_TensorOptions();
  
  if (dtype == "Float") {
    options = lantern_TensorOptions_dtype(options, lantern_Dtype_float64());
    return tensor_to_r_array_double(lantern_Tensor_to(ten, options));
  }
  
  if (dtype == "Long") {
    return tensor_to_r_array_int32_t(ten);
  }
  
  // else if (ten.dtype() == torch::kDouble) {
  //   return tensor_to_r_array<REALSXP, double>(ten);
  // } else if (ten.dtype() == torch::kByte) {
  //   return tensor_to_r_array<LGLSXP, std::uint8_t>(ten);
  // } else if (ten.dtype() == torch::kLong) {
  //   return tensor_to_r_array<INTSXP, int32_t>(ten.to(torch::kInt));
  // } else if (ten.dtype() == torch::kFloat) {
  //   return tensor_to_r_array<REALSXP, double>(ten.to(torch::kDouble));
  // } else if (ten.dtype() == torch::kInt16) {
  //   return tensor_to_r_array<INTSXP, int16_t>(ten.to(torch::kInt16));
  // } else if (ten.dtype() == torch::kBool) {
  //   return tensor_to_r_array<LGLSXP, bool>(ten.to(torch::kBool));
  // }

  Rcpp::stop("dtype not handled");
};

