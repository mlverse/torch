#include "torch_types.h"
#include "utils.h"

void tensor_finalizer (SEXP ptr)
{
  auto xptr = Rcpp::as<Rcpp::XPtr<XPtrTorchTensor>>(ptr);
  lantern_tensor_set_pyobj(xptr->get(), nullptr);
}

XPtrTorchTensor::operator SEXP () const {
  
  // If there's an R object stored in the Tensor Implementation
  // we want to return it directly so we have a unique R object
  // that points to each tensor.
  if (lantern_tensor_get_pyobj(this->get()))
  {
    // It could be that the R objet is still stored in the TensorImpl but
    // it has already been scheduled for finalization by the GC.
    // Thus we need to run the pending finalizers and retry.
    R_RunPendingFinalizers();  
    void* ptr = lantern_tensor_get_pyobj(this->get());
    if (ptr)
    {
      SEXP out = PROTECT(Rf_duplicate((SEXP) ptr));
      UNPROTECT(1);
      return out;
    }
  }
  
  // If there's no R object stored in the Tensor, we will create a new one 
  // and store the weak reference.
  // Since this will be the only R object that points to that tensor, we also
  // register a finalizer that will erase the reference to the R object in the 
  // C++ object whenever this object gets out of scope.
  auto xptr = make_xptr<XPtrTorchTensor>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_tensor", "R7");
  SEXP xptr_ = Rcpp::wrap(xptr);
  R_RegisterCFinalizer(xptr_, tensor_finalizer);
  
  lantern_tensor_set_pyobj(this->get(), (void*) xptr_);
  return xptr_; 
}

XPtrTorchOptionalTensor::operator SEXP() const {
  bool has_value = lantern_optional_tensor_has_value(this->get());
  if (!has_value)
  {
    return R_NilValue;
  }
  else 
  {
    return XPtrTorchTensor(*this);
  }
}

XPtrTorchIndexTensor::operator SEXP () const {
  return XPtrTorchTensor(*this);
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

XPtrTorchGenerator::operator SEXP () const
{
  auto xptr = make_xptr<XPtrTorchGenerator>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_generator", "R7");
  return xptr;
}

XPtrTorchMemoryFormat::operator SEXP () const
{
  auto xptr = make_xptr<XPtrTorchMemoryFormat>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_memory_format", "R7");
  return xptr;
}

// Constructors ----------

XPtrTorchTensor XPtrTorchTensor_from_SEXP (SEXP x)
{
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_tensor")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchTensor>>(x);
    return XPtrTorchTensor( out->get_shared());
  }
  
  if (TYPEOF(x) == NILSXP || (TYPEOF(x) == VECSXP && LENGTH(x) == 0)) {
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

XPtrTorchOptionalTensor XPtrTorchOptionalTensor_from_SEXP (SEXP x)
{
  const bool is_null = TYPEOF(x) == NILSXP || (TYPEOF(x) == VECSXP && LENGTH(x) == 0);
  if (is_null)
  {
    return XPtrTorchOptionalTensor(lantern_optional_tensor(nullptr, true));
  }
  else
  {
    return XPtrTorchOptionalTensor(lantern_optional_tensor(XPtrTorchTensor(x).get(), false));
  }
}

XPtrTorchOptionalTensor::XPtrTorchOptionalTensor (SEXP x) :
  XPtrTorch{XPtrTorchOptionalTensor_from_SEXP(x)} {}

XPtrTorchIndexTensor XPtrTorchIndexTensor_from_SEXP (SEXP x)
{
  XPtrTorchTensor t = XPtrTorchTensor_from_SEXP(x);
  XPtrTorchTensor zero_index = to_index_tensor(t);
  
  return XPtrTorchIndexTensor(zero_index.get_shared());
}

XPtrTorchIndexTensor::XPtrTorchIndexTensor (SEXP x) :
  XPtrTorch{XPtrTorchIndexTensor_from_SEXP(x)} {}

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
  
  if (Rf_isNull(x))
  {
    Rcpp::List tmp; // create an empty list
    return cpp_torch_tensor_list(tmp);
  }
  
  Rcpp::stop("Expected a torch_tensor_list.");
}

XPtrTorchTensorList::XPtrTorchTensorList (SEXP x):
  XPtrTorch{XPtrTorchTensorList_from_SEXP(x)} {}

XPtrTorchOptionalTensorList cpp_torch_optional_tensor_list(const Rcpp::List &x);
XPtrTorchOptionalTensorList XPtrTorchOptionalTensorList_from_SEXP (SEXP x)
{
  
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_tensor"))
  {
    Rcpp::List tmp = Rcpp::List::create(x);
    return cpp_torch_optional_tensor_list(tmp);
  }
  
  if (Rf_isVectorAtomic(x))
  {
    Rcpp::List tmp = Rcpp::List::create(x);
    return cpp_torch_optional_tensor_list(tmp);
  }
  
  if (TYPEOF(x) == VECSXP) 
  {
    return cpp_torch_optional_tensor_list(Rcpp::as<Rcpp::List>(x));
  }
  
  Rcpp::stop("Expected a torch_optional_tensor_list.");
}

XPtrTorchOptionalTensorList::XPtrTorchOptionalTensorList (SEXP x):
  XPtrTorch{XPtrTorchOptionalTensorList_from_SEXP(x)} {}

XPtrTorchIndexTensorList XPtrTorchIndexTensorList_from_SEXP (SEXP x)
{
  XPtrTorchTensorList t = XPtrTorchTensorList_from_SEXP(x);
  XPtrTorchIndexTensorList zero_index = to_index_tensor_list(t);
  
  return zero_index;
}

XPtrTorchIndexTensorList::XPtrTorchIndexTensorList (SEXP x) :
  XPtrTorch{XPtrTorchIndexTensorList_from_SEXP(x)} {}

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
    
    if (args.size() == 0)
    {
      return options;
    }
    
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

XPtrTorchOptionalDevice XPtrTorchOptionalDevice_from_SEXP (SEXP x)
{
  if (TYPEOF(x) == NILSXP)
  {
    return XPtrTorchOptionalDevice(lantern_OptionalDevice_from_device(nullptr, true));
  } 
  else 
  {
    return XPtrTorchOptionalDevice(lantern_OptionalDevice_from_device(XPtrTorchDevice_from_SEXP(x).get(), false));
  }
}

XPtrTorchOptionalDevice::XPtrTorchOptionalDevice (SEXP x):
  XPtrTorch{XPtrTorchOptionalDevice_from_SEXP(x)} {}

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


XPtrTorchGenerator XPtrTorchGenerator_from_SEXP (SEXP x)
{
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_generator"))
  {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchGenerator>>(x);
    return XPtrTorchGenerator( out->get_shared());
  }
  
  if (TYPEOF(x) == NILSXP)
  {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchGenerator>>(
      Rcpp::Environment::namespace_env("torch").find(".generator_null")
    );
    return XPtrTorchGenerator( out->get_shared());
  }
  
  Rcpp::stop("Expected a torch_generator");
}

XPtrTorchGenerator::XPtrTorchGenerator (SEXP x):
  XPtrTorch{XPtrTorchGenerator_from_SEXP(x)} {}

XPtrTorchMemoryFormat XPtrTorchMemoryFormat_from_SEXP (SEXP x)
{
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_memory_format")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchMemoryFormat>>(x);
    return XPtrTorchMemoryFormat( out->get_shared());
  }
  
  if (TYPEOF(x) == NILSXP)
  {
    return XPtrTorchMemoryFormat();
  }
  
  Rcpp::stop("Expected a torch_dtype");
}

XPtrTorchMemoryFormat::XPtrTorchMemoryFormat (SEXP x):
  XPtrTorch{XPtrTorchMemoryFormat_from_SEXP(x)} {}

XPtrTorchIntArrayRef XPtrTorchIntArrayRef_from_SEXP (SEXP x, bool allow_null, bool index)
{
  
  if (TYPEOF(x) == NILSXP)
  {
    if (allow_null)
    {
      return nullptr;  
    } else 
    {
     Rcpp::stop("Expected a list of integers and found NULL."); 
    }
  }
  
  std::vector<int64_t> vec;
  
  if (TYPEOF(x) == VECSXP) 
  {
    auto tmp = Rcpp::as<Rcpp::List>(x);
    for (auto i = tmp.begin(); i != tmp.end(); ++i)
    {
      vec.push_back(Rcpp::as<int64_t>(*i));
    }
  }
  else
  {
    vec = Rcpp::as<std::vector<int64_t>>(x);
  }
  
  if (index)
  {
    for (int i = 0; i < vec.size(); i++)
    {
      
      if (vec[i] == 0)
      {
        Rcpp::stop("Indexing starts at 1 but found a 0.");
      }
      
      if (vec[i] > 0) 
      {
        vec[i] = vec[i] - 1;
      }
    }
  }
  
  auto ptr = lantern_vector_int64_t(vec.data(), vec.size());
  return XPtrTorchIntArrayRef(ptr);
}

XPtrTorchIntArrayRef::XPtrTorchIntArrayRef (SEXP x):
  XPtrTorch{XPtrTorchIntArrayRef_from_SEXP(x, false, false)} {}

XPtrTorchIndexIntArrayRef::XPtrTorchIndexIntArrayRef (SEXP x):
  XPtrTorch{XPtrTorchIntArrayRef_from_SEXP(x, false, true)} {}

XPtrTorchOptionalIntArrayRef XPtrTorchOptionalIntArrayRef_from_SEXP (SEXP x, bool index)
{
  bool is_null;
  std::vector<int64_t> data;
  
  if (TYPEOF(x) == NILSXP || LENGTH(x) == 0)
  {
    is_null = true;
  } 
  else {
    
    // handle lists of integers
    if (TYPEOF(x) == VECSXP)
    {
      auto tmp = Rcpp::as<Rcpp::List>(x);
      for (auto i = tmp.begin(); i != tmp.end(); ++i)
      {
        data.push_back(Rcpp::as<int64_t>(*i));
      }
    }
    else 
    {
      data = Rcpp::as<std::vector<int64_t>>(x);  
    }
    
    
    if (index)
    {
      for (int i = 0; i < data.size(); i++)
      {
        if (data[i] == 0)
        {
          Rcpp::stop("Indexing starts at 1 but found a 0.");
        }
        
        if (data[i] > 0)
        {
          data[i] = data[i] - 1;
        }
        
      }
    }
    
    is_null = false;
  }
  
  return XPtrTorchOptionalIntArrayRef(data, is_null);
}

XPtrTorchOptionalIntArrayRef::XPtrTorchOptionalIntArrayRef (SEXP x):
  XPtrTorchOptionalIntArrayRef{XPtrTorchOptionalIntArrayRef_from_SEXP(x, false)} {};

XPtrTorchOptionalIndexIntArrayRef::XPtrTorchOptionalIndexIntArrayRef (SEXP x):
  XPtrTorchOptionalIntArrayRef{XPtrTorchOptionalIntArrayRef_from_SEXP(x, true)} {};

XPtrTorchint64_t2::XPtrTorchint64_t2 (SEXP x_)
{
  int64_t x;
  if (LENGTH(x_) == 1)
  {
    x = Rcpp::as<int64_t>(x_);  
  } else {
    Rcpp::stop("Expected a single integer.");
  }
  
  ptr = std::shared_ptr<void>(
    lantern_int64_t(x), 
    lantern_int64_t_delete
  );
}

XPtrTorchoptional_int64_t2::XPtrTorchoptional_int64_t2 (SEXP x_)
{
  int64_t x = 0;
  bool is_null;
  
  if (TYPEOF(x_) == NILSXP || LENGTH(x_) == 0)
  {
    is_null = true;
  }
  else
  {
    x = Rcpp::as<int64_t>(x_);  
    is_null = false;
  } 
  ptr = std::shared_ptr<void>(
    lantern_optional_int64_t(x, is_null), 
    lantern_optional_int64_t_delete
  );
}

XPtrTorchindex_int64_t::XPtrTorchindex_int64_t (SEXP x_)
{
  int64_t x = 0;
  
  if (LENGTH(x_) > 0)
  {
    x = Rcpp::as<int64_t>(x_);  
  }
  
  if (x == 0)
  {
    Rcpp::stop("Indexing starts at 1 but found a 0.");
  }
  
  if (x > 0)
  {
    x = x - 1;
  }
  
  ptr = std::shared_ptr<void>(
    lantern_int64_t(x), 
    lantern_int64_t_delete
  );
}

XPtrTorchoptional_index_int64_t::XPtrTorchoptional_index_int64_t (SEXP x_)
{
  int64_t x = 0;
  bool is_null;
  
  if (TYPEOF(x_) == NILSXP || LENGTH(x_) == 0)
  {
    is_null = true;
  }
  else
  {
    x = Rcpp::as<int64_t>(x_);  
    is_null = false;
    if (x == 0)
    {
      Rcpp::stop("Indexing starts at 1 but found a 0.");
    }
    
    if (x > 0)
    {
      x = x - 1;
    }
  }
  
  ptr = std::shared_ptr<void>(
    lantern_optional_int64_t(x, is_null), 
    lantern_optional_int64_t_delete
  );
}

// [[Rcpp::export]]
int test_fun_hello (XPtrTorchOptionalDevice x)
{
  // std::cout << "test fun" << std::endl;
  lantern_print_stuff(x.get());
  return 1 + 1;
}