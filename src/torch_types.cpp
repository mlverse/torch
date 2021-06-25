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

XPtrTorchScriptModule::operator SEXP () const 
{
  
  auto xptr = make_xptr<XPtrTorchScriptModule>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_script_module", "R7");
  
  Rcpp::Function asNamespace("asNamespace");
  Rcpp::Environment torch_pkg = asNamespace("torch");
  Rcpp::Function f = torch_pkg["new_script_module"];
  
  return f(xptr);
}

XPtrTorchScriptMethod::operator SEXP () const 
{
  
  if (!this->get())
  {
    return R_NilValue;
  }
  
  auto xptr = make_xptr<XPtrTorchScriptMethod>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_script_method", "R7");
  
  Rcpp::Function asNamespace("asNamespace");
  Rcpp::Environment torch_pkg = asNamespace("torch");
  Rcpp::Function f = torch_pkg["new_script_method"];
  
  return f(xptr);
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

XPtrTorchvector_string::operator SEXP () const
{
  
  int size = lantern_vector_string_size(this->get());
  
  std::vector<std::string> output;
  for (int i = 0; i < size; i++)
  {
    const char * k = lantern_vector_string_at(this->get(), i);
    output.push_back(std::string(k));
    lantern_const_char_delete(k);
  }
  
  return Rcpp::wrap(output);
}

XPtrTorchstring::operator SEXP () const
{
  const char * out = lantern_string_get(this->get());
  auto output = std::string(out);
  lantern_const_char_delete(out);
  
  return Rcpp::wrap(output);
}

XPtrTorchjit_named_parameter_list::operator SEXP () const 
{
  XPtrTorchTensorList tensors = lantern_jit_named_parameter_list_tensors(this->get());
  XPtrTorchvector_string names = lantern_jit_named_parameter_list_names(this->get());
  Rcpp::List out = Rcpp::wrap(tensors);
  out.attr("names") = Rcpp::wrap(names);
  return out;
}

XPtrTorchjit_named_buffer_list::operator SEXP () const 
{
  XPtrTorchTensorList tensors = lantern_jit_named_buffer_list_tensors(this->get());
  XPtrTorchvector_string names = lantern_jit_named_buffer_list_names(this->get());
  Rcpp::List out = Rcpp::wrap(tensors);
  out.attr("names") = Rcpp::wrap(names);
  return out;
}

XPtrTorchjit_named_module_list::operator SEXP () const 
{
  int size = lantern_jit_named_module_list_size(this->get());
  Rcpp::List out;
  
  if (size == 0)
  {
    return out;
  }
  
  for (int i = 0; i < size; i++)
  {
    out.push_back(XPtrTorchScriptModule(lantern_jit_named_module_list_module_at(this->get(), i)));
  }
  
  XPtrTorchvector_string names = lantern_jit_named_module_list_names(this->get());
  out.attr("names") = Rcpp::wrap(names);
  return out;
}

XPtrTorchvector_bool::operator SEXP() const
{
  int64_t size = lantern_vector_bool_size(this->get());
  std::vector<bool> out;
  for (int i = 0; i < size; i++)
  {
    out.push_back(lantern_vector_bool_at(this->get(), i));
  }
  return Rcpp::wrap(out);
}

XPtrTorchvector_int64_t::operator SEXP() const
{
  int64_t size = lantern_vector_int64_t_size(this->get());
  std::vector<int64_t> out;
  for (int i = 0; i < size; i++)
  {
    out.push_back(lantern_vector_int64_t_at(this->get(), i));
  }
  return Rcpp::wrap(out);
}

XPtrTorchvector_double::operator SEXP() const
{
  double size = lantern_vector_double_size(this->get());
  std::vector<double> out;
  for (int i = 0; i < size; i++)
  {
    out.push_back(lantern_vector_double_at(this->get(), i));
  }
  return Rcpp::wrap(out);
}

XPtrTorchStack::operator SEXP () const 
{

  
}

XPtrTorchIValue::operator SEXP () const
{
  
  int type = lantern_IValue_type(this->get());
  
  switch (type)
  {
  
  case IValue_types::IValueBoolType:
    return Rcpp::wrap(lantern_IValue_Bool(this->get()));
  
  case IValue_types::IValueBoolListType:
    return Rcpp::wrap(XPtrTorchvector_bool(lantern_IValue_BoolList(this->get())));
    
  case IValue_types::IValueDeviceType:
    return Rcpp::wrap(XPtrTorchDevice(lantern_IValue_Device(this->get())));
  
  case IValue_types::IValueDoubleType:
    return Rcpp::wrap(lantern_IValue_Double(this->get()));
    
  case IValue_types::IValueDoubleListType:
    return Rcpp::wrap(XPtrTorchvector_double(lantern_IValue_DoubleList(this->get())));
    
  case IValue_types::IValueGeneratorType:
    return Rcpp::wrap(XPtrTorchGenerator(lantern_IValue_Generator(this->get())));
    
  case IValue_types::IValueIntType:
    return Rcpp::wrap(lantern_IValue_Int(this->get()));
    
  case IValue_types::IValueIntListType:
    return Rcpp::wrap(XPtrTorchvector_int64_t(lantern_IValue_IntList(this->get())));
    
  case IValue_types::IValueModuleType:
    return Rcpp::wrap(XPtrTorchScriptModule(lantern_IValue_Module(this->get())));
    
  case IValue_types::IValueNoneType:
    return R_NilValue;
    
  case IValue_types::IValueScalarType:
    return Rcpp::wrap(XPtrTorchScalar(lantern_IValue_Scalar(this->get())));
    
  case IValue_types::IValueStringType:
    return Rcpp::wrap(XPtrTorchstring(lantern_IValue_String(this->get())));
    
  case IValue_types::IValueTensorListType:
    return Rcpp::wrap(XPtrTorchTensorList(lantern_IValue_TensorList(this->get())));
    
  case IValue_types::IValueTensorType:
    return Rcpp::wrap(XPtrTorchTensor(lantern_IValue_Tensor(this->get())));
    
  case IValue_types::IValueTupleType:
    return Rcpp::wrap(XPtrTorchTuple(lantern_IValue_Tuple(this->get())));
    
  case IValue_types::IValueGenericDictType:
    return Rcpp::wrap(XPtrTorchGenericDict(lantern_IValue_GenericDict(this->get())));
    
  case IValue_types::IValueListType:
    return Rcpp::wrap(XPtrTorchGenericList(lantern_IValue_List(this->get())));
    
  }
  
  Rcpp::Rcout << lantern_IValue_type(this->get()) << std::endl; 
  Rcpp::stop("Type not handled");
}

XPtrTorchTuple::operator SEXP () const 
{
  auto size = lantern_jit_Tuple_size(this->get());
  
  Rcpp::List out;
  for (int i = 0; i < size; i++)
  {
    out.push_back(Rcpp::wrap(XPtrTorchIValue(lantern_jit_Tuple_at(this->get(), i))));
  }
  
  return out;
}

XPtrTorchvector_IValue::operator SEXP () const 
{
  auto size = lantern_jit_vector_IValue_size(this->get());
  
  Rcpp::List out;
  for (int i = 0; i < size; i++)
  {
    out.push_back(Rcpp::wrap(XPtrTorchIValue(lantern_jit_vector_IValue_at(this->get(), i))));
  }
  
  return out;
}

XPtrTorchGenericDict::operator SEXP () const
{
  XPtrTorchvector_IValue keys = lantern_jit_GenericDict_keys(this->get());
  int64_t size = lantern_jit_vector_IValue_size(keys.get());
  
  Rcpp::List out;
  for (int i = 0; i < size; i++)
  {
    out.push_back(
      XPtrTorchIValue(
        lantern_jit_GenericDict_at(
          this->get(),
          XPtrTorchIValue(lantern_jit_vector_IValue_at(keys.get(), i)).get()
        )
      )
    );
  }
  out.attr("names") = Rcpp::wrap(keys);
  return out;
}

XPtrTorchGenericList::operator SEXP () const
{
  
  int64_t size = lantern_jit_GenericList_size(this->get());
  
  Rcpp::List out;
  for (int i = 0; i < size; i++)
  {
    out.push_back(
      XPtrTorchIValue(
        lantern_jit_GenericList_at(
          this->get(),
          i
        )
      )
    );
  }
  
  return out;
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

XPtrTorchScriptModule XPtrTorchScriptModule_from_SEXP (SEXP x)
{
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_script_module")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchScriptModule>>(x);
    return XPtrTorchScriptModule( out->get_shared());
  }
  
  Rcpp::stop("Expected a torch_script_module");
}

XPtrTorchScriptModule::XPtrTorchScriptModule (SEXP x):
  XPtrTorch{XPtrTorchScriptModule_from_SEXP(x)} {}

XPtrTorchScriptMethod XPtrTorchScriptMethod_from_SEXP (SEXP x)
{
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_script_method")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchScriptMethod>>(x);
    return XPtrTorchScriptMethod(out->get_shared());
  }
  
  Rcpp::stop("Expected a torch_script_module");
}

XPtrTorchScriptMethod::XPtrTorchScriptMethod (SEXP x):
  XPtrTorch{XPtrTorchScriptMethod_from_SEXP(x)} {}

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

XPtrTorchstring XPtrTorchstring_from_SEXP (SEXP x)
{
  std::string v = Rcpp::as<std::string>(x);
  return XPtrTorchstring(lantern_string_new(v.c_str()));
}

XPtrTorchstring::XPtrTorchstring(SEXP x) :
  XPtrTorchstring{XPtrTorchstring_from_SEXP(x)} {}

XPtrTorchTuple XPtrTorchTuple_from_SEXP (SEXP x)
{
  auto list = Rcpp::as<Rcpp::List>(x);
  XPtrTorchTuple out = lantern_jit_Tuple_new();
  for (auto el : list)
  {
    lantern_jit_Tuple_push_back(out.get(), Rcpp::as<XPtrTorchIValue>(el).get());
  }
  return out;
}

XPtrTorchTuple::XPtrTorchTuple(SEXP x) :
  XPtrTorchTuple{XPtrTorchTuple_from_SEXP(x)} {}

XPtrTorchTensorDict XPtrTorchTensorDict_from_SEXP (SEXP x)
{
  auto list = Rcpp::as<Rcpp::List>(x);
  XPtrTorchTensorDict out = lantern_jit_TensorDict_new();
  
  Rcpp::List names = list.attr("names");
  for (int i = 0; i < names.size(); i++)
  {
    lantern_jit_TensorDict_push_back(
      out.get(),
      Rcpp::as<XPtrTorchstring>(names[i]).get(),
      Rcpp::as<XPtrTorchTensor>(list[i]).get()
    );
  }
  return out;
}

XPtrTorchTensorDict::XPtrTorchTensorDict(SEXP x) :
  XPtrTorchTensorDict{XPtrTorchTensorDict_from_SEXP(x)} {}

bool rlang_is_named (SEXP x)
{
  auto x_ = Rcpp::as<Rcpp::List>(x);
  SEXP names =  x_.names();
  
  if (Rf_isNull(names)) return false;
  if (x_.size() != Rcpp::as<Rcpp::CharacterVector>(names).size()) return false;
  
  Rcpp::Function asNamespace("asNamespace");
  Rcpp::Environment rlang_pkg = asNamespace("rlang");
  Rcpp::Function f = rlang_pkg["is_named"];
  return f(x);
}

bool is_tensor (SEXP x)
{
  return TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_tensor");
}

XPtrTorchIValue XPtrTorchIValue_from_SEXP (SEXP x)
{
  if (TYPEOF(x) == INTSXP && LENGTH(x) == 1)
  {
    return XPtrTorchIValue(lantern_IValue_from_Int(Rcpp::as<int64_t>(x)));
  }
  
  if (TYPEOF(x) == INTSXP && LENGTH(x) > 1)
  {
    return XPtrTorchIValue(lantern_IValue_from_IntList(Rcpp::as<XPtrTorchvector_int64_t>(x).get()));
  }
  
  if (TYPEOF(x) == CHARSXP && LENGTH(x) == 1)
  {
    return XPtrTorchIValue(lantern_IValue_from_String(Rcpp::as<XPtrTorchstring>(x).get()));
  }
  
  if (TYPEOF(x) == LGLSXP && LENGTH(x) == 1)
  {
    return XPtrTorchIValue(lantern_IValue_from_Bool(Rcpp::as<bool>(x)));
  }
  
  if (TYPEOF(x) == LGLSXP && LENGTH(x) > 1)
  {
    return XPtrTorchIValue(lantern_IValue_from_BoolList(Rcpp::as<XPtrTorchvector_bool>(x).get()));
  }
  
  if (TYPEOF(x) == REALSXP && LENGTH(x) == 1)
  {
    return XPtrTorchIValue(lantern_IValue_from_Double(Rcpp::as<double>(x)));
  }
  
  if (TYPEOF(x) == REALSXP && LENGTH(x) > 1)
  {
    return XPtrTorchIValue(lantern_IValue_from_DoubleList(Rcpp::as<XPtrTorchvector_double>(x).get()));
  }
  
  if (TYPEOF(x)== STRSXP && LENGTH(x) == 1)
  {
    return XPtrTorchIValue(lantern_IValue_from_String(Rcpp::as<XPtrTorchstring>(x).get()));
  }
  
  if (is_tensor(x))
  {
    return XPtrTorchIValue(lantern_IValue_from_Tensor(Rcpp::as<XPtrTorchTensor>(x).get()));
  }
  
  // is a list
  if (TYPEOF(x) == VECSXP)
  {
    
    auto x_ = Rcpp::as<Rcpp::List>(x);
    
    // is a named list, thus we should convert to a dictionary to
    // preserve the names
    if (rlang_is_named(x))
    {
      
      // named list of tensors! we will convert to a Dict
      if (std::all_of(x_.cbegin(), x_.cend(), [](SEXP x){ return is_tensor(x); }))
      {
        return XPtrTorchIValue(lantern_IValue_from_TensorDict(Rcpp::as<XPtrTorchTensorDict>(x).get()));
      }
      
    }
    
    if (std::all_of(x_.cbegin(), x_.cend(), [](SEXP x){ return is_tensor(x); }))
    {
      return XPtrTorchIValue(lantern_IValue_from_TensorList(Rcpp::as<XPtrTorchTensorList>(x).get()));
    }
    
    // it's not a list full of tensors so we need create a tuple
    return XPtrTorchIValue(lantern_IValue_from_Tuple(Rcpp::as<XPtrTorchTuple>(x).get()));
  }
  
  Rcpp::stop("Unsupported type");
}

XPtrTorchIValue::XPtrTorchIValue(SEXP x) :
  XPtrTorchIValue{XPtrTorchIValue_from_SEXP(x)} {};

XPtrTorchvector_bool XPtrTorchvector_bool_from_SEXP (SEXP x)
{
  auto input = Rcpp::as<std::vector<bool>>(x);
  XPtrTorchvector_bool out = lantern_vector_bool_new();
  for (auto el : input)
  {
    lantern_vector_bool_push_back(out.get(), el);
  }
  return out;
}

XPtrTorchvector_bool::XPtrTorchvector_bool (SEXP x) :
  XPtrTorchvector_bool{XPtrTorchvector_bool_from_SEXP(x)} {};

XPtrTorchvector_int64_t XPtrTorchvector_int64_t_from_SEXP (SEXP x)
{
  auto input = Rcpp::as<std::vector<int64_t>>(x);
  XPtrTorchvector_int64_t out = lantern_vector_int64_t_new();
  for (auto el : input)
  {
    lantern_vector_int64_t_push_back(out.get(), el);
  }
  return out;
}

XPtrTorchvector_int64_t::XPtrTorchvector_int64_t (SEXP x) :
  XPtrTorchvector_int64_t{XPtrTorchvector_int64_t_from_SEXP(x)} {};

XPtrTorchvector_double XPtrTorchvector_double_from_SEXP (SEXP x)
{
  auto input = Rcpp::as<std::vector<double>>(x);
  XPtrTorchvector_double out = lantern_vector_double_new();
  for (auto el : input)
  {
    lantern_vector_double_push_back(out.get(), el);
  }
  return out;
}

XPtrTorchvector_double::XPtrTorchvector_double (SEXP x) :
  XPtrTorchvector_double{XPtrTorchvector_double_from_SEXP(x)} {};

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