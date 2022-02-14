
#include <torch.h>

// [[Rcpp::export]]
void cpp_torch_tensor_print(torch::Tensor x, int n) {
  const char* s = lantern_Tensor_StreamInsertion(x.get());
  auto s_string = std::string(s);
  lantern_const_char_delete(
      s);  // above statement has deep copied the s string.

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
  if (cont.size() > n && n > 1) {
    cont.erase(cont.begin() + n, cont.end() - 1);
    truncated = true;
  }

  std::string result;
  for (int i = 0; i < cont.size(); i++) {
    result += cont.at(i);

    if (i != (cont.size() - 1)) result += "\n";

    if (i == (cont.size() - 2) && truncated)
      result += "... [the output was truncated (use n=-1 to disable)]\n";
  }

  Rcpp::Rcout << result;
};

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchDtype> cpp_torch_tensor_dtype(torch::Tensor x) {
  XPtrTorchDtype out = lantern_Tensor_dtype(x.get());
  return make_xptr<XPtrTorchDtype>(out);
}

std::vector<int64_t> stride_from_dim (std::vector<int64_t> x) {
  auto ret = std::vector<int64_t>(x.size());
  ret[0] = 1;
  for (int i = 1; i < x.size(); i++) {
    ret[i] = ret[i-1]*x[i-1];
  }
  return ret;
}

// [[Rcpp::export]]
torch::Tensor torch_tensor_cpp (SEXP x, 
                                Rcpp::Nullable<torch::Dtype> dtype,
                                Rcpp::Nullable<torch::Device> device,
                                bool requires_grad,
                                bool pin_memory) {
  
  torch::Dtype cdtype;
  torch::Dtype final_type;
  
  switch (TYPEOF(x)) {
  case INTSXP: {
    cdtype = lantern_Dtype_int32();
    final_type = dtype.isNull() ? torch::Dtype(lantern_Dtype_int64()) : Rcpp::as<torch::Dtype>(dtype);
    break;
  }
  case REALSXP: {
    auto is_int64 = Rf_inherits(x, "integer64");
    cdtype = is_int64 ? lantern_Dtype_int64() : lantern_Dtype_float64();
    if (dtype.isNull()) {
      final_type = is_int64 ? lantern_Dtype_int64() : lantern_Dtype_float32();
    } else {
      final_type = Rcpp::as<torch::Dtype>(dtype);
    }
    break;
  }
  case LGLSXP: {
    cdtype = lantern_Dtype_int32();
    final_type = dtype.isNull() ? torch::Dtype(lantern_Dtype_bool()) : Rcpp::as<torch::Dtype>(dtype);
    break;
  }
  default: {
    Rcpp::stop("R type not handled");
  }
  }
  
  // We now create the first tensor wrapping the R object. Here we use `cdtype`
  // For example when the SEXP is a logical vector, it actually store values as
  // int32 so we first create a int32 tensor and then cast to a boolean.
  torch::TensorOptions options = lantern_TensorOptions();
  options = lantern_TensorOptions_dtype(options.get(), cdtype.get());
  
  auto d = Rcpp::as<Rcpp::Nullable<std::vector<int64_t>>>(Rf_getAttrib(x, R_DimSymbol));
  auto dim = d.isNotNull() ? Rcpp::as<std::vector<int64_t>>(d) : std::vector<int64_t>(1, LENGTH(x));
  auto strides = stride_from_dim(dim);
  
  torch::Tensor tensor = lantern_from_blob(DATAPTR(x), 
                                           &dim[0], dim.size(), 
                                           &strides[0], strides.size(), 
                                           options.get());
  
  if (dim.size() == 1) {
    // if we have a 1-dim vector contigous doesn't trigger a copy, and
    // would be unexpected.
    tensor = lantern_Tensor_clone(tensor.get());
  }
  tensor = lantern_Tensor_contiguous(tensor.get());
  
  // We will now cast to the final options.
  options = lantern_TensorOptions_dtype(options.get(), final_type.get());
  if (device.isNotNull()) {
    options = lantern_TensorOptions_device(options.get(), Rcpp::as<torch::Device>(device).get());
  }
  options = lantern_TensorOptions_pinned_memory(options.get(), pin_memory);
  tensor = lantern_Tensor_to(tensor.get(), options.get());
  tensor = lantern_Tensor_set_requires_grad(tensor.get(), requires_grad);
  
  return tensor;
}


// A faster version of `torch_stack(lapply(x, torch_tensor), dim = 1)`.
// [[Rcpp::export]]
torch::Tensor stack_list_of_tensors (Rcpp::List x) {
  int n = x.size();
  torch::TensorList out = lantern_TensorList();
  torch::int64_t dim(Rcpp::wrap(0)); 
  for (int i = 0; i < n; i++) {
    auto v = torch_tensor_cpp(x[i]);
    lantern_TensorList_push_back(out.get(), v.get());
  }
  torch::Tensor res = lantern_stack_tensorlist_intt(out.get(), dim.get());
  return res;
}

Rcpp::IntegerVector tensor_dimensions(torch::Tensor x) {
  int64_t ndim = lantern_Tensor_ndimension(x.get());
  Rcpp::IntegerVector dimensions(ndim);
  for (int i = 0; i < ndim; ++i) {
    dimensions[i] = lantern_Tensor_size(x.get(), i);
  }
  return dimensions;
}

Rcpp::List tensor_to_r_array_double(torch::Tensor x) {
  torch::Tensor ten = lantern_Tensor_contiguous(x.get());
  auto d_ptr = lantern_Tensor_data_ptr_double(ten.get());
  Rcpp::Vector<REALSXP> vec(d_ptr, d_ptr + lantern_Tensor_numel(ten.get()));
  return Rcpp::List::create(Rcpp::Named("vec") = vec,
                            Rcpp::Named("dim") = tensor_dimensions(x));
}

Rcpp::List tensor_to_r_array_uint8_t(torch::Tensor x) {
  torch::Tensor ten = lantern_Tensor_contiguous(x.get());
  auto d_ptr = lantern_Tensor_data_ptr_uint8_t(ten.get());
  Rcpp::Vector<LGLSXP> vec(d_ptr, d_ptr + lantern_Tensor_numel(ten.get()));
  return Rcpp::List::create(Rcpp::Named("vec") = vec,
                            Rcpp::Named("dim") = tensor_dimensions(x));
}

Rcpp::List tensor_to_r_array_int32_t(torch::Tensor x) {
  torch::Tensor ten = lantern_Tensor_contiguous(x.get());
  auto d_ptr = lantern_Tensor_data_ptr_int32_t(ten.get());
  Rcpp::Vector<INTSXP> vec(d_ptr, d_ptr + lantern_Tensor_numel(ten.get()));
  return Rcpp::List::create(Rcpp::Named("vec") = vec,
                            Rcpp::Named("dim") = tensor_dimensions(x));
}

Rcpp::List tensor_to_r_array_int64_t(torch::Tensor x) {
  torch::Tensor ten = lantern_Tensor_contiguous(x.get());
  auto d_ptr = lantern_Tensor_data_ptr_int64_t(ten.get());

  int64_t len = lantern_Tensor_numel(ten.get());
  Rcpp::NumericVector vec(len);  // storage vehicle we return them in

  // transfers values 'keeping bits' but changing type
  // using reinterpret_cast would get us a warning
  std::memcpy(&(vec[0]), d_ptr, len * sizeof(double));

  vec.attr("class") = "integer64";
  return Rcpp::List::create(Rcpp::Named("vec") = vec,
                            Rcpp::Named("dim") = tensor_dimensions(x));
}

Rcpp::List tensor_to_r_array_bool(torch::Tensor x) {
  torch::Tensor ten = lantern_Tensor_contiguous(x.get());
  auto d_ptr = lantern_Tensor_data_ptr_bool(ten.get());
  Rcpp::Vector<LGLSXP> vec(d_ptr, d_ptr + lantern_Tensor_numel(ten.get()));
  return Rcpp::List::create(Rcpp::Named("vec") = vec,
                            Rcpp::Named("dim") = tensor_dimensions(x));
}

// [[Rcpp::export]]
Rcpp::List cpp_as_array(Rcpp::XPtr<torch::Tensor> x) {
  auto s =
      lantern_Dtype_type(XPtrTorchDtype(lantern_Tensor_dtype(x->get())).get());
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

  torch::TensorOptions options = lantern_TensorOptions();

  if (dtype == "Float") {
    options = lantern_TensorOptions_dtype(
        options.get(), XPtrTorchDtype(lantern_Dtype_float64()).get());
    return tensor_to_r_array_double(
        torch::Tensor(lantern_Tensor_to(x->get(), options.get())));
  }

  if (dtype == "Long") {
    return tensor_to_r_array_int64_t(*x.get());
  }

  Rcpp::stop("dtype not handled");
};

// [[Rcpp::export]]
int cpp_tensor_element_size(Rcpp::XPtr<torch::Tensor> x) {
  return lantern_Tensor_element_size(x->get());
}

// [[Rcpp::export]]
std::vector<int> cpp_tensor_dim(Rcpp::XPtr<torch::Tensor> x) {
  auto ndim = lantern_Tensor_ndimension(x->get());
  std::vector<int> out;
  for (int i = 0; i < ndim; i++) {
    out.push_back(lantern_Tensor_size(x->get(), i));
  }
  return out;
}

// [[Rcpp::export]]
int cpp_tensor_numel(Rcpp::XPtr<torch::Tensor> x) {
  return lantern_Tensor_numel(x->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchDevice> cpp_tensor_device(Rcpp::XPtr<torch::Tensor> self) {
  XPtrTorchDevice out = lantern_Tensor_device(self->get());
  return make_xptr<XPtrTorchDevice>(out);
}

// [[Rcpp::export]]
bool cpp_tensor_is_undefined(Rcpp::XPtr<torch::Tensor> self) {
  return lantern_Tensor_is_undefined(self->get());
}

// [[Rcpp::export]]
bool cpp_tensor_is_contiguous(Rcpp::XPtr<torch::Tensor> self) {
  return lantern_Tensor_is_contiguous(self->get());
}

// [[Rcpp::export]]
bool cpp_tensor_has_names(Rcpp::XPtr<torch::Tensor> self) {
  return lantern_Tensor_has_names(self->get());
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchDimnameList> cpp_tensor_names(
    Rcpp::XPtr<torch::Tensor> self) {
  return make_xptr<XPtrTorchDimnameList>(lantern_Tensor_names(self->get()));
}

// [[Rcpp::export]]
void cpp_set_num_threads(int n) { lantern_set_num_threads(n); }

// [[Rcpp::export]]
void cpp_set_num_interop_threads(int n) { lantern_set_num_interop_threads(n); }

// [[Rcpp::export]]
int cpp_get_num_threads() { return lantern_get_num_threads(); }

// [[Rcpp::export]]
int cpp_get_num_interop_threads() { return lantern_get_num_interop_threads(); }

// [[Rcpp::export]]
torch::Tensor cpp_namespace_normal_double_double(
    double mean, double std, std::vector<int64_t> size,
    Rcpp::XPtr<XPtrTorchGenerator> generator, torch::TensorOptions options) {
  torch::Tensor out =
      lantern_normal_double_double_intarrayref_generator_tensoroptions(
          mean, std,
          XPtrTorchvector_int64_t(
              lantern_vector_int64_t(size.data(), size.size()))
              .get(),
          generator->get(), options.get());
  return out;
}

// [[Rcpp::export]]
torch::Tensor cpp_namespace_normal_double_tensor(
    double mean, Rcpp::XPtr<torch::Tensor> std,
    Rcpp::XPtr<XPtrTorchGenerator> generator) {
  torch::Tensor out = lantern_normal_double_tensor_generator(mean, std->get(),
                                                             generator->get());
  return out;
}

// [[Rcpp::export]]
torch::Tensor cpp_namespace_normal_tensor_double(
    Rcpp::XPtr<torch::Tensor> mean, double std,
    Rcpp::XPtr<XPtrTorchGenerator> generator) {
  torch::Tensor out = lantern_normal_tensor_double_generator(mean->get(), std,
                                                             generator->get());
  return out;
}

// [[Rcpp::export]]
torch::Tensor cpp_namespace_normal_tensor_tensor(
    Rcpp::XPtr<torch::Tensor> mean, Rcpp::XPtr<torch::Tensor> std,
    Rcpp::XPtr<XPtrTorchGenerator> generator) {
  torch::Tensor out = lantern_normal_tensor_tensor_generator(
      mean->get(), std->get(), generator->get());
  return out;
}

// [[Rcpp::export]]
torch::Tensor nnf_pad_circular(torch::Tensor input,
                               XPtrTorchIntArrayRef padding) {
  return torch::Tensor(
      lantern_nn_functional_pad_circular(input.get(), padding.get()));
}
