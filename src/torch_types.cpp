#include "torch_types.h"
#include "utils.h"

XPtrTorchTensor::operator SEXP () const {
  auto xptr = make_xptr<XPtrTorchTensor>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_tensor", "R7");
  return xptr; 
}

XPtrTorchTensorList::operator SEXP () const {
  Rcpp::List out;
  int64_t sze = lantern_TensorList_size(this->get());
  
  for (int i = 0; i < sze; i++)
  {
    void * tmp = lantern_TensorList_at(this->get(), i);
    out.push_back(XPtrTorchTensor(tmp));
  }
  
  return out;
}

XPtrTorchScalarType::operator SEXP () const {
  auto xptr = make_xptr<XPtrTorchScalarType>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_dtype", "R7");
  return xptr; 
}

XPtrTorchScalar::operator SEXP () const {
  XPtrTorchScalarType dtype_ptr = lantern_Scalar_dtype(this->get());
  const char * dtype_c = lantern_Dtype_type(dtype_ptr.get());
  auto dtype = std::string(dtype_c);
  lantern_const_char_delete(dtype_c);
  
  Rcpp::RObject output;
  if (dtype == "Double") {
    output = lantern_Scalar_to_double(this->get());
  } else if (dtype == "Float") {
    output = lantern_Scalar_to_float(this->get());
  } else if (dtype == "Bool") {
    output = lantern_Scalar_to_bool(this->get());
  } else if (dtype == "Int") {
    output = lantern_Scalar_to_int(this->get());
  } else if (dtype == "Long") {
    output = lantern_Scalar_to_int(this->get());
  } else {
    Rcpp::stop("Cannot convert from scalar of type.");
  }
  
  return output; 
}

XPtrTorchTensorOptions::operator SEXP () const 
{
  auto xptr = make_xptr<XPtrTorchTensorOptions>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_tensor_options", "R7");
  return xptr;
}

XPtrTorchDevice::operator SEXP () const 
{
  auto xptr = make_xptr<XPtrTorchDevice>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_device", "R7");
  return xptr;
}

XPtrTorchDtype::operator SEXP () const 
{
  auto xptr = make_xptr<XPtrTorchDtype>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_dtype", "R7");
  return xptr;
}

XPtrTorchDimname::operator SEXP () const
{
  auto xptr = make_xptr<XPtrTorchDimname>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_dimname", "R7");
  return xptr;
}

XPtrTorchDimnameList::operator SEXP () const
{
  auto xptr = make_xptr<XPtrTorchDimnameList>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_dimname_list", "R7");
  return xptr;
}

// Constructors ----------

XPtrTorchTensor XPtrTorchTensor_from_SEXP (SEXP x)
{
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_tensor")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchTensor>>(x);
    return XPtrTorchTensor( out->get_shared());
  }
  
  if (TYPEOF(x) == NILSXP) {
    return cpp_tensor_undefined();
  }
  
  // TODO: it would be nice to make it all C++ 
  if (Rf_isVectorAtomic(x)) {
    Rcpp::Environment torch_pkg = Rcpp::Environment("package:torch");
    Rcpp::Function f = torch_pkg["torch_tensor"];
    return XPtrTorchTensor(Rcpp::as<Rcpp::XPtr<XPtrTorchTensor>>(f(x))->get_shared());
  }
  
  Rcpp::stop("Expected a torch_tensor.");
}

XPtrTorchTensor::XPtrTorchTensor (SEXP x) : 
  XPtrTorch{XPtrTorchTensor_from_SEXP(x)} {}


XPtrTorchScalar cpp_torch_scalar (SEXP x);
XPtrTorchScalar XPtrTorchScalar_from_SEXP (SEXP x)
{
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_scalar")) 
  {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchScalar>>(x);
    return XPtrTorchScalar( out->get_shared());
  }
  
  if (Rf_isVectorAtomic(x) && (Rf_length(x) == 1))
  {
    return cpp_torch_scalar(x);
  }
  
  if (TYPEOF(x) == NILSXP) {
    return XPtrTorchScalar();
  }
  
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_tensor"))
  {
    auto ten = Rcpp::as<Rcpp::XPtr<XPtrTorchTensor>>(x);
    return XPtrTorchScalar(lantern_Tensor_item_tensor(ten->get()));
  }
  
  Rcpp::stop("Expected a torch_scalar.");
}

XPtrTorchScalar::XPtrTorchScalar (SEXP x):
  XPtrTorch{XPtrTorchScalar_from_SEXP(x)} {}

XPtrTorchTensorList cpp_torch_tensor_list(const Rcpp::List &x);
XPtrTorchTensorList XPtrTorchTensorList_from_SEXP (SEXP x)
{
  
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_tensor_list")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchTensorList>>(x);
    return XPtrTorchTensorList( out->get_shared());
  }
  
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_tensor"))
  {
    Rcpp::List tmp = Rcpp::List::create(x);
    return cpp_torch_tensor_list(tmp);
  }
  
  if (Rf_isVectorAtomic(x))
  {
    Rcpp::List tmp = Rcpp::List::create(x);
    return cpp_torch_tensor_list(tmp);
  }
  
  if (TYPEOF(x) == VECSXP) 
  {
    return cpp_torch_tensor_list(Rcpp::as<Rcpp::List>(x));
  }
  
  Rcpp::stop("Expected a torch_tensor_list.");
}

XPtrTorchTensorList::XPtrTorchTensorList (SEXP x):
  XPtrTorch{XPtrTorchTensorList_from_SEXP(x)} {}

XPtrTorchTensorOptions XPtrTorchTensorOptions_from_SEXP (SEXP x)
{
  
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_tensor_options")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchTensorOptions>>(x);
    return XPtrTorchTensorOptions( out->get_shared());
  }
  
  if (TYPEOF(x) == VECSXP || Rf_inherits(x, "torch_tensor_options")) 
  {
    XPtrTorchTensorOptions options(lantern_TensorOptions());
    Rcpp::List args = Rcpp::as<Rcpp::List>(x);
    std::vector<std::string> names = args.names();
    
    for (auto i = names.begin(); i != names.end(); ++i)
    {
      if (TYPEOF(args[*i]) == NILSXP) 
      {
        continue;
      }
      
      if (*i == "dtype")
      {
        auto dtype = *Rcpp::as<Rcpp::XPtr<XPtrTorch>>(args[*i]);
        options = lantern_TensorOptions_dtype(options.get(), dtype.get());
      }
      if (*i == "layout") {
        auto layout = *Rcpp::as<Rcpp::XPtr<XPtrTorch>>(args[*i]);
        options = lantern_TensorOptions_layout(options.get(), layout.get());
      }
      if (*i == "device") {
        auto device = * Rcpp::as<Rcpp::XPtr<XPtrTorch>>(args[*i]);
        options = lantern_TensorOptions_device(options.get(), device.get());
      }
      if (*i == "requires_grad") {
        options = lantern_TensorOptions_requires_grad(options.get(), Rcpp::as<bool>(args[*i]));
      }
      if (*i == "pinned_memory") {
        options = lantern_TensorOptions_pinned_memory(options.get(), Rcpp::as<bool>(args[*i]));
      }
    }
    
    return options;
  }
  
  Rcpp::stop("Expected a torch_tensor_option.");
}

XPtrTorchTensorOptions::XPtrTorchTensorOptions (SEXP x):
  XPtrTorch{XPtrTorchTensorOptions_from_SEXP(x)} {}

XPtrTorchDevice cpp_torch_device(std::string type, Rcpp::Nullable<std::int64_t> index);
XPtrTorchDevice XPtrTorchDevice_from_SEXP (SEXP x)
{
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_device")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchDevice>>(x);
    return XPtrTorchDevice( out->get_shared());
  }
  
  if (TYPEOF(x) == VECSXP && Rf_inherits(x, "torch_device"))
  {
    auto a = Rcpp::as<Rcpp::List>(x);
    return cpp_torch_device(
      Rcpp::as<std::string>(a["type"]), 
      Rcpp::as<Rcpp::Nullable<std::int64_t>>(a["index"])
    );
  }
  
  if (TYPEOF(x) == STRSXP && (LENGTH(x) == 1))
  {
    auto str = Rcpp::as<std::string>(x);
    SEXP index = R_NilValue;
    
    auto delimiter = str.find(":");
    if (delimiter!=std::string::npos) {
      index = Rcpp::wrap(std::stoi(str.substr(delimiter + 1, str.length())));
      str = str.substr(0, delimiter);
    }
    
    return cpp_torch_device(str, index);
  }
  
  Rcpp::stop("Expected a torch_device");
}

XPtrTorchDevice::XPtrTorchDevice (SEXP x):
  XPtrTorch{XPtrTorchDevice_from_SEXP(x)} {}

XPtrTorchDtype XPtrTorchDtype_from_SEXP (SEXP x)
{
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_dtype")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchDtype>>(x);
    return XPtrTorchDtype( out->get_shared());
  }
  
  if (TYPEOF(x) == NILSXP)
  {
    return XPtrTorchDtype();
  }
  
  Rcpp::stop("Expected a torch_dtype");
}

XPtrTorchDtype::XPtrTorchDtype (SEXP x):
  XPtrTorch{XPtrTorchDtype_from_SEXP(x)} {}

XPtrTorchDimname XPtrTorchDimname_from_SEXP (SEXP x)
{
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_dimname"))
  {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchDimname>>(x);
    return XPtrTorchDimname( out->get_shared());
  }
  
  if (TYPEOF(x) == STRSXP && (LENGTH(x) == 1))
  {
    return XPtrTorchDimname(Rcpp::as<std::string>(x));
  }
  
  Rcpp::stop("Expected a torch_dimname");
}

XPtrTorchDimname::XPtrTorchDimname (SEXP x):
  XPtrTorch{XPtrTorchDimname_from_SEXP(x)} {}

XPtrTorchDimnameList XPtrTorchDimnameList_from_SEXP (SEXP x)
{
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_dimname_list"))
  {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchDimnameList>>(x);
    return XPtrTorchDimnameList( out->get_shared());
  }
  
  if (TYPEOF(x) == STRSXP)
  {
    XPtrTorchDimnameList out = lantern_DimnameList();
    auto names = Rcpp::as<std::vector<std::string>>(x);
    for (int i = 0; i < names.size(); i++) {
      lantern_DimnameList_push_back(out.get(), XPtrTorchDimname(names[i]).get());
    }
    return out;
  }
  
  Rcpp::stop("Expected a torch_dimname_list");
}

XPtrTorchDimnameList::XPtrTorchDimnameList (SEXP x):
  XPtrTorch{XPtrTorchDimnameList_from_SEXP(x)} {}


// [[Rcpp::export]]
[[gnu::noinline]]
XPtrTorchDimname test_fun (XPtrTorchDimname x)
{
  return x;
}

// [[Rcpp::export]]
[[gnu::noinline]]
XPtrTorchDimnameList test_fun2 (XPtrTorchDimnameList x)
{
  return x;
}