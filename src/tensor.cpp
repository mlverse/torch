
#include "torch_types.h"
#include "utils.h"

// [[Rcpp::export]]
void cpp_torch_tensor_print (Rcpp::XPtr<XPtrTorchTensor> x, int n) {
  const char* s = lantern_Tensor_StreamInsertion(x->get());
  auto s_string = std::string(s);
  lantern_const_char_delete(s); // above statement has deep copied the s string.
  
  // https://stackoverflow.com/a/55742744/3297472
  // split string into lines without using streams as they are 
  // not supported in older gcc versions
  std::vector<std::string> cont;
  size_t start = 0;
  size_t end;
  while (1) {
    std::string token;
    if ((end = s_string.find("\n", start)) == std::string::npos) {
      
      if (!(token = s_string.substr(start)).empty()) {
        cont.push_back(token);
      }
       
      break;
    }
     
    token = s_string.substr(start, end - start);
    cont.push_back(token);
    start = end + 1;
  }
  
  bool truncated = false;
  if (cont.size() > n && n > 1)
  {
    cont.erase(cont.begin() + n, cont.end() - 1);
    truncated = true;
  }
  
  std::string result;
  for (int i = 0; i < cont.size(); i ++)
  {
    result += cont.at(i);
    
    if (i != (cont.size() - 1))
      result += "\n";
    
    if (i == (cont.size() - 2) && truncated)
      result += "... [the output was truncated (use n=-1 to disable)]\n";
  }
  
  Rcpp::Rcout << result << std::endl;
};

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchDtype> cpp_torch_tensor_dtype(Rcpp::XPtr<XPtrTorchTensor> x) {
  XPtrTorchDtype out = lantern_Tensor_dtype(x->get());
  return make_xptr<XPtrTorchDtype>(out);
}

// equivalent to (n-1):0 in R
std::vector<int64_t> revert_int_seq (int n) {
  std::vector<int64_t> l(n);
  std::iota(l.begin(), l.end(), 0);
  std::reverse(l.begin(), l.end());
  return l;
};

template <int RTYPE>
XPtrTorchTensor tensor_from_r_array (const SEXP x, std::vector<int64_t> dim, std::string dtype) {

  Rcpp::Vector<RTYPE> vec(x);

  XPtrTorchTensorOptions options = lantern_TensorOptions();
  
  if (dtype == "double") 
  {
    options = lantern_TensorOptions_dtype(options.get(), XPtrTorchDtype(lantern_Dtype_float64()).get());
  } 
  else if (dtype == "int") 
  {
    options = lantern_TensorOptions_dtype(options.get(), XPtrTorchDtype(lantern_Dtype_int32()).get());
  } 
  else if (dtype == "int64")
  {
    options = lantern_TensorOptions_dtype(options.get(), XPtrTorchDtype(lantern_Dtype_int64()).get());
  }

  options = lantern_TensorOptions_device(options.get(), XPtrTorchDevice(lantern_Device("cpu", 0, false)).get());

  XPtrTorchTensor tensor = lantern_from_blob(vec.begin(), &dim[0], dim.size(), options.get());

  if (dim.size() == 1) {
    // if we have a 1-dim vector contigous doesn't trigger a copy, and
    // would be unexpected.
    tensor = lantern_Tensor_clone(tensor.get());
  }

  auto reverse_dim = revert_int_seq(dim.size());
  tensor = lantern_Tensor_permute(tensor.get(), XPtrTorchvector_int64_t(lantern_vector_int64_t(&reverse_dim[0], reverse_dim.size())).get());
  tensor = lantern_Tensor_contiguous(tensor.get());

  return tensor;
};

// [[Rcpp::export]]
XPtrTorchTensor cpp_torch_tensor (SEXP x, std::vector<std::int64_t> dim,
                                  XPtrTorchTensorOptions options,
                                  bool requires_grad, bool is_integer64) {
  
  XPtrTorchTensor tensor;

  if (TYPEOF(x) == INTSXP) 
  {
    tensor = tensor_from_r_array<INTSXP>(x, dim, "int");
  } 
  else if (TYPEOF(x) == REALSXP && !is_integer64) 
  {
    tensor = tensor_from_r_array<REALSXP>(x, dim, "double");
  } 
  else if (TYPEOF(x) == REALSXP && is_integer64)
  {
    tensor = tensor_from_r_array<REALSXP>(x, dim, "int64");
  }
  else if (TYPEOF(x) == LGLSXP) 
  {
    tensor = tensor_from_r_array<LGLSXP>(x, dim, "int");
  } 
  else 
  {
    Rcpp::stop("R type not handled");
  };
  
  tensor = lantern_Tensor_to(tensor.get(), options.get());
  tensor = lantern_Tensor_set_requires_grad(tensor.get(), requires_grad);

  return tensor;
}

Rcpp::IntegerVector tensor_dimensions (XPtrTorchTensor x) {
  int64_t ndim = lantern_Tensor_ndimension(x.get());
  Rcpp::IntegerVector dimensions(ndim);
  for (int i = 0; i < ndim; ++i) {
    dimensions[i] = lantern_Tensor_size(x.get(), i);
  }
  return dimensions;
}

Rcpp::List tensor_to_r_array_double (XPtrTorchTensor x) {
  XPtrTorchTensor ten = lantern_Tensor_contiguous(x.get());
  auto d_ptr = lantern_Tensor_data_ptr_double(ten.get());
  Rcpp::Vector<REALSXP> vec(d_ptr, d_ptr + lantern_Tensor_numel(ten.get()));
  return Rcpp::List::create(Rcpp::Named("vec") = vec, Rcpp::Named("dim") = tensor_dimensions(x));
}

Rcpp::List tensor_to_r_array_uint8_t (XPtrTorchTensor x) {
  XPtrTorchTensor ten = lantern_Tensor_contiguous(x.get());
  auto d_ptr = lantern_Tensor_data_ptr_uint8_t(ten.get());
  Rcpp::Vector<LGLSXP> vec(d_ptr, d_ptr + lantern_Tensor_numel(ten.get()));
  return Rcpp::List::create(Rcpp::Named("vec") = vec, Rcpp::Named("dim") = tensor_dimensions(x));
}

Rcpp::List tensor_to_r_array_int32_t (XPtrTorchTensor x) {
  XPtrTorchTensor ten = lantern_Tensor_contiguous(x.get());
  auto d_ptr = lantern_Tensor_data_ptr_int32_t(ten.get());
  Rcpp::Vector<INTSXP> vec(d_ptr, d_ptr + lantern_Tensor_numel(ten.get()));
  return Rcpp::List::create(Rcpp::Named("vec") = vec, Rcpp::Named("dim") = tensor_dimensions(x));
}

Rcpp::List tensor_to_r_array_int64_t (XPtrTorchTensor x)
{
  XPtrTorchTensor ten = lantern_Tensor_contiguous(x.get());
  auto d_ptr = lantern_Tensor_data_ptr_int64_t(ten.get());
  
  int64_t len = lantern_Tensor_numel(ten.get());
  Rcpp::NumericVector vec(len);         // storage vehicle we return them in
  
  // transfers values 'keeping bits' but changing type
  // using reinterpret_cast would get us a warning
  std::memcpy(&(vec[0]), d_ptr, len * sizeof(double));
  
  vec.attr("class") = "integer64"; 
  return Rcpp::List::create(Rcpp::Named("vec") = vec, Rcpp::Named("dim") = tensor_dimensions(x));
}

Rcpp::List tensor_to_r_array_bool (XPtrTorchTensor x) {
  XPtrTorchTensor ten = lantern_Tensor_contiguous(x.get());
  auto d_ptr = lantern_Tensor_data_ptr_bool(ten.get());
  Rcpp::Vector<LGLSXP> vec(d_ptr, d_ptr + lantern_Tensor_numel(ten.get()));
  return Rcpp::List::create(Rcpp::Named("vec") = vec, Rcpp::Named("dim") = tensor_dimensions(x));
}

// [[Rcpp::export]]
Rcpp::List cpp_as_array (Rcpp::XPtr<XPtrTorchTensor> x) {
  
  auto s = lantern_Dtype_type(XPtrTorchDtype(lantern_Tensor_dtype(x->get())).get());
  auto dtype = std::string(s);
  lantern_const_char_delete(s);
  
  if (dtype == "Byte") {
    return tensor_to_r_array_uint8_t(*x.get());
  } 
  
  if (dtype == "Int") {
    return tensor_to_r_array_int32_t(*x.get());
  }
  
  if (dtype == "Bool") {
    return tensor_to_r_array_bool(*x.get());
  }
  
  if (dtype == "Double") {
    return tensor_to_r_array_double(*x.get());
  }
  
  XPtrTorchTensorOptions options = lantern_TensorOptions();
  
  if (dtype == "Float") {
    options = lantern_TensorOptions_dtype(options.get(), XPtrTorchDtype(lantern_Dtype_float64()).get());
    return tensor_to_r_array_double(XPtrTorchTensor(lantern_Tensor_to(x->get(), options.get())));
  }
  
  if (dtype == "Long") {
    return tensor_to_r_array_int64_t(*x.get());
  }
  
  Rcpp::stop("dtype not handled");
};

// [[Rcpp::export]]
int cpp_tensor_element_size (Rcpp::XPtr<XPtrTorchTensor> x) {
  return lantern_Tensor_element_size(x->get());
}

// [[Rcpp::export]]
std::vector<int> cpp_tensor_dim (Rcpp::XPtr<XPtrTorchTensor> x) {
  auto ndim = lantern_Tensor_ndimension(x->get());
  std::vector<int> out;
  for (int i = 0; i < ndim; i++) {
    out.push_back(lantern_Tensor_size(x->get(), i));
  }
  return out;
}

// [[Rcpp::export]]
int cpp_tensor_numel (Rcpp::XPtr<XPtrTorchTensor> x) {
  return lantern_Tensor_numel(x->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchDevice> cpp_tensor_device (Rcpp::XPtr<XPtrTorchTensor> self) {
  XPtrTorchDevice out = lantern_Tensor_device(self->get());
  return make_xptr<XPtrTorchDevice>(out);
}

// [[Rcpp::export]]
bool cpp_tensor_is_undefined (Rcpp::XPtr<XPtrTorchTensor> self)
{
  return lantern_Tensor_is_undefined(self->get());
}

// [[Rcpp::export]]
bool cpp_tensor_is_contiguous (Rcpp::XPtr<XPtrTorchTensor> self)
{
  return lantern_Tensor_is_contiguous(self->get());
}

// [[Rcpp::export]]
bool cpp_tensor_has_names (Rcpp::XPtr<XPtrTorchTensor> self)
{
  return lantern_Tensor_has_names(self->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchDimnameList> cpp_tensor_names (Rcpp::XPtr<XPtrTorchTensor> self)
{
  return make_xptr<XPtrTorchDimnameList>(lantern_Tensor_names(self->get()));
}

// [[Rcpp::export]]
void cpp_set_num_threads (int n)
{
  lantern_set_num_threads(n);
}

// [[Rcpp::export]]
void cpp_set_num_interop_threads (int n)
{
  lantern_set_num_interop_threads(n);
}

// [[Rcpp::export]]
int cpp_get_num_threads ()
{
  return lantern_get_num_threads();
}

// [[Rcpp::export]]
int cpp_get_num_interop_threads ()
{
  return lantern_get_num_interop_threads();
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_namespace_normal_double_double (double mean, double std, 
                                                                std::vector<int64_t> size,
                                                                Rcpp::XPtr<XPtrTorchGenerator> generator,
                                                                XPtrTorchTensorOptions options) {
  XPtrTorchTensor out = lantern_normal_double_double_intarrayref_generator_tensoroptions(
    mean, std, 
    XPtrTorchvector_int64_t(lantern_vector_int64_t(size.data(), size.size())).get(),
    generator->get(),
    options.get()
  );
  return out;
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_namespace_normal_double_tensor (
    double mean, 
    Rcpp::XPtr<XPtrTorchTensor> std,
    Rcpp::XPtr<XPtrTorchGenerator> generator
)
{
  XPtrTorchTensor out = lantern_normal_double_tensor_generator(
    mean, 
    std->get(), 
    generator->get()
  );
  return out; 
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_namespace_normal_tensor_double (
    Rcpp::XPtr<XPtrTorchTensor> mean,
    double std, 
    Rcpp::XPtr<XPtrTorchGenerator> generator
)
{
  XPtrTorchTensor out = lantern_normal_tensor_double_generator(
    mean->get(), 
    std, 
    generator->get()
  );
  return out; 
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_namespace_normal_tensor_tensor (
    Rcpp::XPtr<XPtrTorchTensor> mean,
    Rcpp::XPtr<XPtrTorchTensor> std, 
    Rcpp::XPtr<XPtrTorchGenerator> generator
)
{
  XPtrTorchTensor out = lantern_normal_tensor_tensor_generator(
    mean->get(), 
    std->get(), 
    generator->get()
  );
  return out; 
}

// [[Rcpp::export]]
XPtrTorchTensor nnf_pad_circular (XPtrTorchTensor input, XPtrTorchIntArrayRef padding)
{
  return XPtrTorchTensor(lantern_nn_functional_pad_circular(input.get(), padding.get()));
}

