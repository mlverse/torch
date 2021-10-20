#include <torch.h>
#include "utils.h"
#include <algorithm>

// [[Rcpp::export]]
Rcpp::XPtr<std::nullptr_t> cpp_nullptr () {
  return make_xptr<std::nullptr_t>(nullptr);
}

// [[Rcpp::export]]
Rcpp::XPtr<std::nullptr_t> cpp_nullopt () {
  return make_xptr<std::nullptr_t>(nullptr);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchoptional_int64_t> cpp_optional_int64_t (Rcpp::Nullable<int64_t> x)
{
  XPtrTorchoptional_int64_t out = nullptr;
  if (x.isNull()) {
    out = lantern_optional_int64_t(0, true);
  } else {
    out = lantern_optional_int64_t(Rcpp::as<int64_t>(x), false);
  }
  return make_xptr<XPtrTorchoptional_int64_t>(out);
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_tensor_undefined () {
  return XPtrTorchTensor(lantern_Tensor_undefined());
}

XPtrTorchTensor to_index_tensor (XPtrTorchTensor t) 
{
  // check that there's no zeros
  bool zeros = lantern_Tensor_has_any_zeros(t.get());
  if (zeros)
  {
    Rcpp::stop("Indexing starts at 1 but found a 0.");
  }
  
  /// make it 0 based!
  XPtrTorchTensor sign = lantern_Tensor_signbit_tensor(t.get());
  sign = lantern_logical_not_tensor(sign.get());
  
  // cast from bool to int
  XPtrTorchTensorOptions options = lantern_TensorOptions();
  options = lantern_TensorOptions_dtype(options.get(), XPtrTorchDtype(lantern_Dtype_int64()).get());
  sign = lantern_Tensor_to(sign.get(), options.get());
  
  // create a 1 scalar
  int al = 1;
  XPtrTorchScalar alpha = lantern_Scalar((void*) &al, std::string("int").c_str());
  XPtrTorchTensor zero_index = lantern_Tensor_sub_tensor_tensor_scalar(
    t.get(), 
    sign.get(), 
    alpha.get()
  );
  
  return zero_index;
}

XPtrTorchIndexTensorList to_index_tensor_list (XPtrTorchTensorList x)
{
  XPtrTorchIndexTensorList out = lantern_TensorList();
  int64_t sze = lantern_TensorList_size(x.get());
  
  for (int i=0; i < sze; i++)
  {
    XPtrTorchTensor t = lantern_TensorList_at(x.get(), i);
    XPtrTorchTensor zero_index = to_index_tensor (t);
    lantern_TensorList_push_back(out.get(), zero_index.get());
  }
  
  return out;
}

XPtrTorchOptionalIndexTensorList to_optional_index_tensor_list (XPtrTorchOptionalTensorList x)
{
  XPtrTorchOptionalIndexTensorList out = lantern_OptionalTensorList();
  int64_t sze = lantern_OptionalTensorList_size(x.get());
  
  for (int i=0; i < sze; i++)
  {
    if (lantern_OptionalTensorList_at_is_null(x.get(), i))
    {
      lantern_OptionalTensorList_push_back(out.get(), nullptr, true);
    }
    else
    {
      XPtrTorchTensor t = lantern_OptionalTensorList_at(x.get(), i); 
      XPtrTorchTensor zero_index = to_index_tensor (t);
      lantern_OptionalTensorList_push_back(out.get(), zero_index.get(), false);
    }
  }
  
  return out;
}

// [[Rcpp::export]]
bool cpp_torch_namespace__use_cudnn_rnn_flatten_weight ()
{
  return lantern__use_cudnn_rnn_flatten_weight();
}
