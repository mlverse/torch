#define TORCH_IMPL
#include <torch.h>

// This file defines torch functions that are exported in order to create
// torch's C API. Most of functions here should not be directly used, but they
// are called by torch type wrappers.

static inline void tensor_finalizer(SEXP ptr) {
  auto xptr = Rcpp::as<Rcpp::XPtr<XPtrTorchTensor>>(ptr);
  lantern_tensor_set_pyobj(xptr->get(), nullptr);
}

static inline bool rlang_is_named(SEXP x) {
  auto x_ = Rcpp::as<Rcpp::List>(x);
  SEXP names = x_.names();

  if (Rf_isNull(names)) return false;
  if (x_.size() != Rcpp::as<Rcpp::CharacterVector>(names).size()) return false;

  Rcpp::Function asNamespace("asNamespace");
  Rcpp::Environment rlang_pkg = asNamespace("rlang");
  Rcpp::Function f = rlang_pkg["is_named"];
  return f(x);
}

static inline bool is_tensor(SEXP x) {
  return TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_tensor");
}

static inline XPtrTorchIndexTensorList to_index_tensor_list(
    XPtrTorchTensorList x) {
  XPtrTorchIndexTensorList out = lantern_TensorList();
  int64_t sze = lantern_TensorList_size(x.get());

  for (int i = 0; i < sze; i++) {
    XPtrTorchTensor t = lantern_TensorList_at(x.get(), i);
    XPtrTorchTensor zero_index = to_index_tensor(t);
    lantern_TensorList_push_back(out.get(), zero_index.get());
  }

  return out;
}

static inline XPtrTorchOptionalTensorList cpp_torch_optional_tensor_list(
    const Rcpp::List& x) {
  XPtrTorchOptionalTensorList out = lantern_OptionalTensorList();

  SEXP item;
  for (int i = 0; i < x.length(); i++) {
    item = x.at(i);
    if (Rf_isNull(item)) {
      lantern_OptionalTensorList_push_back(out.get(), nullptr, true);
    } else {
      lantern_OptionalTensorList_push_back(out.get(),
                                           XPtrTorchTensor(item).get(), false);
    }
  }

  return out;
}

static inline XPtrTorchOptionalIndexTensorList to_optional_index_tensor_list(
    XPtrTorchOptionalTensorList x) {
  XPtrTorchOptionalIndexTensorList out = lantern_OptionalTensorList();
  int64_t sze = lantern_OptionalTensorList_size(x.get());

  for (int i = 0; i < sze; i++) {
    if (lantern_OptionalTensorList_at_is_null(x.get(), i)) {
      lantern_OptionalTensorList_push_back(out.get(), nullptr, true);
    } else {
      XPtrTorchTensor t = lantern_OptionalTensorList_at(x.get(), i);
      XPtrTorchTensor zero_index = to_index_tensor(t);
      lantern_OptionalTensorList_push_back(out.get(), zero_index.get(), false);
    }
  }

  return out;
}

XPtrTorchTensorList cpp_torch_tensor_list(const Rcpp::List& x);
XPtrTorchScalar cpp_torch_scalar(SEXP x);
XPtrTorchOptionalTensorList cpp_torch_optional_tensor_list(const Rcpp::List& x);
XPtrTorchDevice cpp_torch_device(std::string type,
                                 Rcpp::Nullable<std::int64_t> index);

// torch_tensor

SEXP operator_sexp_tensor(const XPtrTorchTensor* self) {
  // If there's an R object stored in the Tensor Implementation
  // we want to return it directly so we have a unique R object
  // that points to each tensor.
  if (lantern_tensor_get_pyobj(self->get())) {
    // It could be that the R objet is still stored in the TensorImpl but
    // it has already been scheduled for finalization by the GC.
    // Thus we need to run the pending finalizers and retry.
    R_RunPendingFinalizers();
    void* ptr = lantern_tensor_get_pyobj(self->get());
    if (ptr) {
      SEXP out = PROTECT(Rf_duplicate((SEXP)ptr));
      UNPROTECT(1);
      return out;
    }
  }

  // If there's no R object stored in the Tensor, we will create a new one
  // and store the weak reference.
  // Since this will be the only R object that points to that tensor, we also
  // register a finalizer that will erase the reference to the R object in the
  // C++ object whenever this object gets out of scope.
  auto xptr = make_xptr<XPtrTorchTensor>(*self);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_tensor", "R7");
  SEXP xptr_ = PROTECT(Rcpp::wrap(xptr));
  R_RegisterCFinalizer(xptr_, tensor_finalizer);
  lantern_tensor_set_pyobj(self->get(), (void*)xptr_);
  UNPROTECT(1);
  return xptr_;
}

XPtrTorchTensor from_sexp_tensor(SEXP x) {
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_tensor")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchTensor>>(x);
    return XPtrTorchTensor(out->get_shared());
  }

  if (TYPEOF(x) == NILSXP || (TYPEOF(x) == VECSXP && LENGTH(x) == 0)) {
    return cpp_tensor_undefined();
  }

  if (Rf_isVectorAtomic(x)) {
    return torch_tensor_cpp(x);
  }

  Rcpp::stop("Expected a torch_tensor.");
}

void delete_tensor(void* x) { lantern_Tensor_delete(x); }

// optional_torch_tensor

SEXP operator_sexp_optional_tensor(const XPtrTorchOptionalTensor* self) {
  if (!lantern_optional_tensor_has_value(self->get())) {
    return R_NilValue;
  }

  return Rcpp::wrap(
      XPtrTorchTensor(lantern_optional_tensor_value(self->get())));
}

XPtrTorchOptionalTensor from_sexp_optional_tensor(SEXP x) {
  const bool is_null =
      TYPEOF(x) == NILSXP || (TYPEOF(x) == VECSXP && LENGTH(x) == 0);
  if (is_null) {
    return XPtrTorchOptionalTensor(lantern_optional_tensor(nullptr));
  } else {
    return XPtrTorchOptionalTensor(
        lantern_optional_tensor(Rcpp::as<XPtrTorchTensor>(x).get()));
  }
}

void delete_optional_tensor(void* x) { lantern_optional_tensor_delete(x); }

// index tensor

XPtrTorchIndexTensor from_sexp_index_tensor(SEXP x) {
  XPtrTorchTensor t = from_sexp_tensor(x);
  XPtrTorchTensor zero_index = to_index_tensor(t);

  return XPtrTorchIndexTensor(zero_index.get_shared());
}

// tensor_list

SEXP operator_sexp_tensor_list(const XPtrTorchTensorList* self) {
  Rcpp::List out;
  int64_t sze = lantern_TensorList_size(self->get());

  for (int i = 0; i < sze; i++) {
    void* tmp = lantern_TensorList_at(self->get(), i);
    out.push_back(XPtrTorchTensor(tmp));
  }

  return out;
}

XPtrTorchTensorList from_sexp_tensor_list(SEXP x) {
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_tensor_list")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchTensorList>>(x);
    return XPtrTorchTensorList(out->get_shared());
  }

  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_tensor")) {
    Rcpp::List tmp = Rcpp::List::create(x);
    return cpp_torch_tensor_list(tmp);
  }

  if (Rf_isVectorAtomic(x)) {
    Rcpp::List tmp = Rcpp::List::create(x);
    return cpp_torch_tensor_list(tmp);
  }

  if (TYPEOF(x) == VECSXP) {
    return cpp_torch_tensor_list(Rcpp::as<Rcpp::List>(x));
  }

  if (Rf_isNull(x)) {
    Rcpp::List tmp;  // create an empty list
    return cpp_torch_tensor_list(tmp);
  }

  Rcpp::stop("Expected a torch_tensor_list.");
}

void delete_tensor_list(void* x) { lantern_TensorList_delete(x); }

// scalar

SEXP operator_sexp_scalar(const XPtrTorchScalar* self) {
  XPtrTorchScalarType dtype_ptr = lantern_Scalar_dtype(self->get());
  const char* dtype_c = lantern_Dtype_type(dtype_ptr.get());
  auto dtype = std::string(dtype_c);
  lantern_const_char_delete(dtype_c);

  Rcpp::RObject output;
  if (dtype == "Double") {
    output = lantern_Scalar_to_double(self->get());
  } else if (dtype == "Float") {
    output = lantern_Scalar_to_float(self->get());
  } else if (dtype == "Bool") {
    output = lantern_Scalar_to_bool(self->get());
  } else if (dtype == "Int") {
    output = lantern_Scalar_to_int(self->get());
  } else if (dtype == "Long") {
    output = lantern_Scalar_to_int(self->get());
  } else {
    Rcpp::stop("Cannot convert from scalar of type.");
  }

  return output;
}

XPtrTorchScalar from_sexp_scalar(SEXP x) {
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_scalar")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchScalar>>(x);
    return XPtrTorchScalar(out->get_shared());
  }

  if (Rf_isVectorAtomic(x) && (Rf_length(x) == 1)) {
    return cpp_torch_scalar(x);
  }

  if (TYPEOF(x) == NILSXP) {
    return XPtrTorchScalar();
  }

  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_tensor")) {
    auto ten = Rcpp::as<Rcpp::XPtr<XPtrTorchTensor>>(x);
    return XPtrTorchScalar(lantern_Tensor_item_tensor(ten->get()));
  }

  Rcpp::stop("Expected a torch_scalar.");
}

void delete_scalar(void* x) { lantern_Scalar_delete(x); }

// optional scalar

SEXP operator_sexp_optional_scalar(const XPtrTorchoptional_scalar* x) {
  if (!lantern_optional_scalar_has_value(x->get())) {
    return R_NilValue;
  }

  return XPtrTorchScalar(lantern_optional_scalar_value(x->get()));
}

XPtrTorchoptional_scalar from_sexp_optional_scalar(SEXP x) {
  if (TYPEOF(x) == NILSXP) {
    return XPtrTorchoptional_scalar(lantern_optional_scalar(nullptr));
  }

  return XPtrTorchoptional_scalar(
      lantern_optional_scalar(Rcpp::as<XPtrTorchScalar>(x).get()));
}

void delete_optional_scalar(void* x) { lantern_optional_scalar_delete(x); }

// scalar type

SEXP operator_sexp_scalar_type(const XPtrTorchScalarType* self) {
  auto xptr = make_xptr<XPtrTorchScalarType>(*self);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_dtype", "R7");
  return xptr;
}

XPtrTorchScalarType from_sexp_scalar_type(SEXP x) {
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_dtype")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchScalarType>>(x);
    return XPtrTorchScalarType(out->get_shared());
  }

  Rcpp::stop("Expected a scalar type");
}

void delete_scalar_type(void* x) { lantern_ScalarType_delete(x); }

// optional scalar type

SEXP operator_sexp_optional_scalar_type(
    const XPtrTorchoptional_scalar_type* self) {
  if (!lantern_optional_scalar_type_has_value(self->get())) {
    return R_NilValue;
  }

  return XPtrTorchoptional_scalar_type(
      XPtrTorchScalarType(lantern_optional_scalar_type_value(self->get()))
          .get());
}

XPtrTorchoptional_scalar_type from_sexp_optional_scalar_type(SEXP x) {
  if (TYPEOF(x) == NILSXP) {
    return XPtrTorchoptional_scalar_type(lantern_optional_scalar_type(nullptr));
  }

  return XPtrTorchoptional_scalar_type(
      lantern_optional_scalar_type(Rcpp::as<XPtrTorchScalarType>(x).get()));
}

void delete_optional_scalar_type(void* x) {
  lantern_optional_scalar_type_delete(x);
}

// tensor options

SEXP operator_sexp_tensor_options(const XPtrTorchTensorOptions* self) {
  auto xptr = make_xptr<XPtrTorchTensorOptions>(*self);
  xptr.attr("class") =
      Rcpp::CharacterVector::create("torch_tensor_options", "R7");
  return xptr;
}

XPtrTorchTensorOptions from_sexp_tensor_options(SEXP x) {
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_tensor_options")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchTensorOptions>>(x);
    return XPtrTorchTensorOptions(out->get_shared());
  }

  if (TYPEOF(x) == VECSXP || Rf_inherits(x, "torch_tensor_options")) {
    XPtrTorchTensorOptions options(lantern_TensorOptions());
    Rcpp::List args = Rcpp::as<Rcpp::List>(x);

    if (args.size() == 0) {
      return options;
    }

    std::vector<std::string> names = args.names();

    for (auto i = names.begin(); i != names.end(); ++i) {
      if (TYPEOF(args[*i]) == NILSXP) {
        continue;
      }

      if (*i == "dtype") {
        auto dtype = *Rcpp::as<Rcpp::XPtr<XPtrTorch>>(args[*i]);
        options = lantern_TensorOptions_dtype(options.get(), dtype.get());
      }
      if (*i == "layout") {
        auto layout = *Rcpp::as<Rcpp::XPtr<XPtrTorch>>(args[*i]);
        options = lantern_TensorOptions_layout(options.get(), layout.get());
      }
      if (*i == "device") {
        auto device = *Rcpp::as<Rcpp::XPtr<XPtrTorch>>(args[*i]);
        options = lantern_TensorOptions_device(options.get(), device.get());
      }
      if (*i == "requires_grad") {
        options = lantern_TensorOptions_requires_grad(options.get(),
                                                      Rcpp::as<bool>(args[*i]));
      }
      if (*i == "pinned_memory") {
        options = lantern_TensorOptions_pinned_memory(options.get(),
                                                      Rcpp::as<bool>(args[*i]));
      }
    }

    return options;
  }

  Rcpp::stop("Expected a torch_tensor_option.");
}

void delete_tensor_options(void* x) { lantern_TensorOptions_delete(x); }

// compilation unit

SEXP operator_sexp_compilation_unit(const XPtrTorchCompilationUnit* self) {
  auto xptr = make_xptr<XPtrTorchCompilationUnit>(*self);
  xptr.attr("class") =
      Rcpp::CharacterVector::create("torch_compilation_unit", "R7");
  return xptr;
}

XPtrTorchCompilationUnit from_sexp_compilation_unit(SEXP x) {
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_compilation_unit")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchCompilationUnit>>(x);
    return XPtrTorchCompilationUnit(out->get_shared());
  }
  Rcpp::stop("Unsupported type. Expected an external pointer.");
}

void delete_compilation_unit(void* x) { lantern_CompilationUnit_delete(x); }

// device

SEXP operator_sexp_device(const XPtrTorchDevice* self) {
  auto xptr = make_xptr<XPtrTorchDevice>(*self);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_device", "R7");
  return xptr;
}

XPtrTorchDevice from_sexp_device(SEXP x) {
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_device")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchDevice>>(x);
    return XPtrTorchDevice(out->get_shared());
  }

  if (TYPEOF(x) == VECSXP && Rf_inherits(x, "torch_device")) {
    auto a = Rcpp::as<Rcpp::List>(x);
    return cpp_torch_device(Rcpp::as<std::string>(a["type"]),
                            Rcpp::as<Rcpp::Nullable<std::int64_t>>(a["index"]));
  }

  if (TYPEOF(x) == STRSXP && (LENGTH(x) == 1)) {
    auto str = Rcpp::as<std::string>(x);
    SEXP index = R_NilValue;

    auto delimiter = str.find(":");
    if (delimiter != std::string::npos) {
      index = Rcpp::wrap(std::stoi(str.substr(delimiter + 1, str.length())));
      str = str.substr(0, delimiter);
    }

    return cpp_torch_device(str, index);
  }

  Rcpp::stop("Expected a torch_device");
}

void delete_device(void* x) { lantern_Device_delete(x); }

// script module

SEXP operator_sexp_script_module(const XPtrTorchScriptModule* self) {
  auto xptr = make_xptr<XPtrTorchScriptModule>(*self);
  xptr.attr("class") =
      Rcpp::CharacterVector::create("torch_script_module", "R7");

  Rcpp::Function asNamespace("asNamespace");
  Rcpp::Environment torch_pkg = asNamespace("torch");
  Rcpp::Function f = torch_pkg["new_script_module"];

  return f(xptr);
}

XPtrTorchScriptModule from_sexp_script_module(SEXP x) {
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_script_module")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchScriptModule>>(x);
    return XPtrTorchScriptModule(out->get_shared());
  }

  if (x == R_NilValue) {
    return XPtrTorchScriptModule((void*)nullptr);
  }

  Rcpp::stop("Expected a torch_script_module");
}

void delete_script_module(void* x) { lantern_JITModule_delete(x); }

// script method

SEXP operator_sexp_script_method(const XPtrTorchScriptMethod* self) {
  if (!self->get()) {
    return R_NilValue;
  }

  auto xptr = make_xptr<XPtrTorchScriptMethod>(*self);
  xptr.attr("class") =
      Rcpp::CharacterVector::create("torch_script_method", "R7");

  Rcpp::Function asNamespace("asNamespace");
  Rcpp::Environment torch_pkg = asNamespace("torch");
  Rcpp::Function f = torch_pkg["new_script_method"];

  return f(xptr);
}

XPtrTorchScriptMethod from_sexp_script_method(SEXP x) {
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_script_method")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchScriptMethod>>(x);
    return XPtrTorchScriptMethod(out->get_shared());
  }

  Rcpp::stop("Expected a torch_script_module");
}

void delete_script_method(void* x) { lantern_jit_ScriptMethod_delete(x); }

// dtype

SEXP operator_sexp_dtype(const XPtrTorchDtype* self) {
  auto xptr = make_xptr<XPtrTorchDtype>(*self);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_dtype", "R7");
  return xptr;
}

XPtrTorchDtype from_sexp_dtype(SEXP x) {
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_dtype")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchDtype>>(x);
    return XPtrTorchDtype(out->get_shared());
  }

  if (TYPEOF(x) == NILSXP) {
    return XPtrTorchDtype();
  }

  Rcpp::stop("Expected a torch_dtype");
}

void delete_dtype(void* x) { lantern_Dtype_delete(x); }

// dimname

SEXP operator_sexp_dimname(const XPtrTorchDimname* self) {
  auto xptr = make_xptr<XPtrTorchDimname>(*self);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_dimname", "R7");
  return xptr;
}

XPtrTorchDimname from_sexp_dimname(SEXP x) {
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_dimname")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchDimname>>(x);
    return XPtrTorchDimname(out->get_shared());
  }

  if (TYPEOF(x) == STRSXP && (LENGTH(x) == 1)) {
    auto str = Rcpp::as<XPtrTorchstring>(x);
    return XPtrTorchDimname(lantern_Dimname(str.get()));
  }

  Rcpp::stop("Expected a torch_dimname");
}

void delete_dimname(void* x) { lantern_Dimname_delete(x); }

// dimname_list

SEXP operator_sexp_dimname_list(const XPtrTorchDimnameList* self) {
  auto xptr = make_xptr<XPtrTorchDimnameList>(*self);
  xptr.attr("class") =
      Rcpp::CharacterVector::create("torch_dimname_list", "R7");
  return xptr;
}

XPtrTorchDimnameList from_sexp_dimname_list(SEXP x) {
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_dimname_list")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchDimnameList>>(x);
    return XPtrTorchDimnameList(out->get_shared());
  }

  if (TYPEOF(x) == STRSXP) {
    XPtrTorchDimnameList out = lantern_DimnameList();
    auto names = Rcpp::as<std::vector<std::string>>(x);
    for (int i = 0; i < names.size(); i++) {
      lantern_DimnameList_push_back(out.get(),
                                    XPtrTorchDimname(names[i]).get());
    }
    return out;
  }

  Rcpp::stop("Expected a torch_dimname_list");
}

void delete_dimname_list(void* x) { lantern_DimnameList_delete(x); }

// Optional dimname list

SEXP operator_sexp_optional_dimname_list(
    const XPtrTorchOptionalDimnameList* self) {
  if (!lantern_optional_dimname_list_has_value(self->get())) {
    return R_NilValue;
  }

  return Rcpp::wrap(
      XPtrTorchDimnameList(lantern_optional_dimname_list_value(self->get())));
}

XPtrTorchOptionalDimnameList from_sexp_optional_dimname_list(SEXP x) {
  if (TYPEOF(x) == NILSXP) {
    return XPtrTorchOptionalDimnameList(lantern_optional_dimname_list(nullptr));
  } else {
    return XPtrTorchOptionalDimnameList(
        lantern_optional_dimname_list(Rcpp::as<XPtrTorchDimnameList>(x).get()));
  }
}

void delete_optional_dimname_list(void* x) {
  lantern_optional_dimname_list_delete(x);
}

// generator

SEXP operator_sexp_generator(const XPtrTorchGenerator* self) {
  auto xptr = make_xptr<XPtrTorchGenerator>(*self);
  xptr.attr("class") = Rcpp::CharacterVector::create("torch_generator", "R7");
  return xptr;
}

XPtrTorchGenerator from_sexp_generator(SEXP x) {
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_generator")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchGenerator>>(x);
    return XPtrTorchGenerator(out->get_shared());
  }

  if (TYPEOF(x) == NILSXP) {
    Rcpp::Function torch_option =
        Rcpp::Environment::namespace_env("torch").find("torch_option");

    if (Rcpp::as<bool>(torch_option("old_seed_behavior", false))) {
      auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchGenerator>>(
          Rcpp::Environment::namespace_env("torch").find(".generator_null"));
      return XPtrTorchGenerator(out->get_shared());
    } else {
      return XPtrTorchGenerator(lantern_get_default_Generator());
    }
  }

  Rcpp::stop("Expected a torch_generator");
}

void delete_generator(void* x) { lantern_Generator_delete(x); }

// optional generator

SEXP operator_sexp_optional_generator(const XPtrTorchOptionalGenerator* self) {
  if (!lantern_optional_generator_has_value(self->get())) {
    return R_NilValue;
  }

  return Rcpp::wrap(
      XPtrTorchGenerator(lantern_optional_generator_value(self->get())));
}

XPtrTorchOptionalGenerator from_sexp_optional_generator(SEXP x) {
  // We actualy do the same as we do with non-optional Generator's which is
  // getting the default LibTorch Generator and pass that.
  // This is because we want to have full control over the default Generator
  // to be able to make changes that don't break backward compatibility.
  return XPtrTorchOptionalGenerator(
      lantern_optional_generator(Rcpp::as<XPtrTorchGenerator>(x).get()));
}

void delete_optional_generator(void* x) {
  lantern_optional_generator_delete(x);
}

// memory format

SEXP operator_sexp_memory_format(const XPtrTorchMemoryFormat* self) {
  auto xptr = make_xptr<XPtrTorchMemoryFormat>(*self);
  xptr.attr("class") =
      Rcpp::CharacterVector::create("torch_memory_format", "R7");
  return xptr;
}

XPtrTorchMemoryFormat from_sexp_memory_format(SEXP x) {
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_memory_format")) {
    auto out = Rcpp::as<Rcpp::XPtr<XPtrTorchMemoryFormat>>(x);
    return XPtrTorchMemoryFormat(out->get_shared());
  }

  if (TYPEOF(x) == NILSXP) {
    return XPtrTorchMemoryFormat();
  }

  Rcpp::stop("Expected a torch_dtype");
}

void delete_memory_format(void* x) { lantern_MemoryFormat_delete(x); }

// optional mmory format

SEXP operator_sexp_optional_memory_format(
    const XPtrTorchoptional_memory_format* x) {
  if (!lantern_optional_memory_format_has_value(x->get())) {
    return R_NilValue;
  }

  return XPtrTorchMemoryFormat(lantern_optional_memory_format_value(x->get()));
}

XPtrTorchoptional_memory_format from_sexp_optional_memory_format(SEXP x) {
  if (TYPEOF(x) == NILSXP) {
    return XPtrTorchoptional_memory_format(
        lantern_optional_memory_format(nullptr));
  }

  return XPtrTorchoptional_memory_format(
      lantern_optional_memory_format(Rcpp::as<XPtrTorchMemoryFormat>(x).get()));
}

void delete_optional_memory_format(void* x) {
  lantern_optional_memory_format_delete(x);
}

// vector string

SEXP operator_sexp_vector_string(const XPtrTorchvector_string* self) {
  int size = lantern_vector_string_size(self->get());

  std::vector<std::string> output;
  for (int i = 0; i < size; i++) {
    const char* k = lantern_vector_string_at(self->get(), i);
    output.push_back(std::string(k));
    lantern_const_char_delete(k);
  }

  return Rcpp::wrap(output);
}

XPtrTorchvector_string from_sexp_vector_string(SEXP x) {
  XPtrTorchvector_string out = lantern_vector_string_new();
  auto strings = Rcpp::as<std::vector<std::string>>(x);
  for (int i = 0; i < strings.size(); i++) {
    lantern_vector_string_push_back(out.get(), strings.at(i).c_str());
  }
  return out;
}

void delete_vector_string(void* x) { lantern_vector_string_delete(x); }

// vector scalar

SEXP operator_sexp_vector_scalar(const XPtrTorchvector_Scalar* self) {
  int size = lantern_vector_Scalar_size(self->get());
  Rcpp::List output;
  for (int i = 0; i < size; i++) {
    auto value = XPtrTorchScalar(lantern_vector_Scalar_at(self->get(), i));
    output.push_back(value);
  }
  return output;
}

XPtrTorchvector_Scalar from_sexp_vector_scalar(SEXP x) {
  auto input = Rcpp::as<Rcpp::List>(x);
  XPtrTorchvector_Scalar output = lantern_vector_Scalar_new();
  for (auto& el : input) {
    lantern_vector_Scalar_push_back(output.get(),
                                    Rcpp::as<XPtrTorchScalar>(el));
  }
  return output;
}

void delete_vector_scalar(void* x) { lantern_vector_Scalar_delete(x); }

// string

SEXP operator_sexp_string(const XPtrTorchstring* self) {
  const char* out = lantern_string_get(self->get());
  auto output = std::string(out);
  lantern_const_char_delete(out);

  return Rcpp::wrap(output);
}

XPtrTorchstring from_sexp_string(SEXP x) {
  std::string v = Rcpp::as<std::string>(x);
  return XPtrTorchstring(lantern_string_new(v.c_str()));
}

void delete_string(void* x) { lantern_string_delete(x); }

void* fixme_new_string(const char* x) { return lantern_string_new(x); }

// string_view

XPtrTorchstring_view from_sexp_string_view(SEXP x) {
  std::string v = Rcpp::as<std::string>(x);
  return XPtrTorchstring_view(lantern_string_view_new(v.c_str()));
}

void delete_string_view(void* x) { lantern_string_view_delete(x); }

// optional string view

XPtrTorchoptional_string_view from_sexp_optional_string_view(SEXP x) {
  if (TYPEOF(x) == NILSXP) {
    return XPtrTorchoptional_string_view(lantern_optional_string_view(nullptr));
  }

  return XPtrTorchoptional_string_view(
      lantern_optional_string_view(Rcpp::as<XPtrTorchstring_view>(x).get()));
}

void delete_optional_string_view(void* x) {
  lantern_optional_string_view_delete(x);
}

// optional string

SEXP operator_sexp_optional_string(const XPtrTorchoptional_string* x) {
  if (!lantern_optional_string_has_value(x->get())) {
    return R_NilValue;
  }

  return XPtrTorchstring(lantern_optional_string_value(x->get()));
}

XPtrTorchoptional_string from_sexp_optional_string(SEXP x) {
  if (TYPEOF(x) == NILSXP) {
    return XPtrTorchoptional_string(lantern_optional_string(nullptr));
  }

  return XPtrTorchoptional_string(
      lantern_optional_string(Rcpp::as<XPtrTorchstring>(x).get()));
}

void delete_optional_string(void* x) { lantern_optional_string_delete(x); }

// jit_named_parameter_list

SEXP operator_sexp_jit_named_parameter_list(
    const XPtrTorchjit_named_parameter_list* self) {
  XPtrTorchTensorList tensors =
      lantern_jit_named_parameter_list_tensors(self->get());
  XPtrTorchvector_string names =
      lantern_jit_named_parameter_list_names(self->get());
  Rcpp::List out = Rcpp::wrap(tensors);
  out.attr("names") = Rcpp::wrap(names);
  return out;
}

void delete_jit_named_parameter_list(void* x) {
  lantern_jit_named_parameter_list_delete(x);
}

// jit_named_buffer_list

SEXP operator_sexp_jit_named_buffer_list(
    const XPtrTorchjit_named_buffer_list* self) {
  XPtrTorchTensorList tensors =
      lantern_jit_named_buffer_list_tensors(self->get());
  XPtrTorchvector_string names =
      lantern_jit_named_buffer_list_names(self->get());
  Rcpp::List out = Rcpp::wrap(tensors);
  out.attr("names") = Rcpp::wrap(names);
  return out;
}

void delete_jit_named_buffer_list(void* x) {
  lantern_jit_named_buffer_list_delete(x);
}

// named_modules_list

SEXP operator_sexp_jit_named_module_list(
    const XPtrTorchjit_named_module_list* self) {
  int size = lantern_jit_named_module_list_size(self->get());
  Rcpp::List out;

  if (size == 0) {
    return out;
  }

  for (int i = 0; i < size; i++) {
    out.push_back(XPtrTorchScriptModule(
        lantern_jit_named_module_list_module_at(self->get(), i)));
  }

  XPtrTorchvector_string names =
      lantern_jit_named_module_list_names(self->get());
  out.attr("names") = Rcpp::wrap(names);
  return out;
}

void delete_jit_named_module_list(void* x) {
  lantern_jit_named_module_list_delete(x);
}

// vector_bool

SEXP operator_sexp_vector_bool(const XPtrTorchvector_bool* self) {
  int64_t size = lantern_vector_bool_size(self->get());
  std::vector<bool> out;
  for (int i = 0; i < size; i++) {
    out.push_back(lantern_vector_bool_at(self->get(), i));
  }
  return Rcpp::wrap(out);
}

XPtrTorchvector_bool from_sexp_vector_bool(SEXP x) {
  auto input = Rcpp::as<std::vector<bool>>(x);
  XPtrTorchvector_bool out = lantern_vector_bool_new();
  for (auto el : input) {
    lantern_vector_bool_push_back(out.get(), el);
  }
  return out;
}

void delete_vector_bool(void* x) { lantern_vector_bool_delete(x); }

// vector_inpt64_t

SEXP operator_sexp_vector_int64_t(const XPtrTorchvector_int64_t* self) {
  int64_t size = lantern_vector_int64_t_size(self->get());
  std::vector<int64_t> out;
  for (int i = 0; i < size; i++) {
    out.push_back(lantern_vector_int64_t_at(self->get(), i));
  }
  return Rcpp::wrap(out);
}

XPtrTorchvector_int64_t from_sexp_vector_int64_t(SEXP x) {
  auto input = Rcpp::as<std::vector<int64_t>>(x);
  XPtrTorchvector_int64_t out = lantern_vector_int64_t_new();
  for (auto el : input) {
    lantern_vector_int64_t_push_back(out.get(), el);
  }
  return out;
}

void delete_vector_int64_t(void* x) { lantern_vector_int64_t_delete(x); }

// vector_double

SEXP operator_sexp_vector_double(const XPtrTorchvector_double* self) {
  double size = lantern_vector_double_size(self->get());
  std::vector<double> out;
  for (int i = 0; i < size; i++) {
    out.push_back(lantern_vector_double_at(self->get(), i));
  }
  return Rcpp::wrap(out);
}

XPtrTorchvector_double from_sexp_vector_double(SEXP x) {
  auto input = Rcpp::as<std::vector<double>>(x);
  XPtrTorchvector_double out = lantern_vector_double_new();
  for (auto el : input) {
    lantern_vector_double_push_back(out.get(), el);
  }
  return out;
}

void delete_vector_double(void* x) { lantern_vector_double_delete(x); }

// stack

SEXP operator_sexp_stack(const XPtrTorchStack* self) {
  int64_t size = lantern_Stack_size(self->get());
  Rcpp::List output;
  for (int64_t i = 0; i < size; i++) {
    output.push_back(XPtrTorchIValue(lantern_Stack_at(self->get(), i)));
  }
  return output;
}

XPtrTorchStack from_sexp_stack(SEXP x) {
  auto list = Rcpp::as<Rcpp::List>(x);
  XPtrTorchStack output = lantern_Stack_new();
  for (const auto& el : list) {
    lantern_Stack_push_back_IValue(output.get(),
                                   Rcpp::as<XPtrTorchIValue>(el).get());
  }
  return output;
}

void delete_stack(void* x) { lantern_Stack_delete(x); }

// ivalue

SEXP operator_sexp_ivalue(const XPtrTorchIValue* self) {
  int type = lantern_IValue_type(self->get());

  switch (type) {
    case IValue_types::IValueBoolType:
      return Rcpp::wrap(lantern_IValue_Bool(self->get()));

    case IValue_types::IValueBoolListType:
      return Rcpp::wrap(
          XPtrTorchvector_bool(lantern_IValue_BoolList(self->get())));

    case IValue_types::IValueDeviceType:
      return Rcpp::wrap(XPtrTorchDevice(lantern_IValue_Device(self->get())));

    case IValue_types::IValueDoubleType:
      return Rcpp::wrap(lantern_IValue_Double(self->get()));

    case IValue_types::IValueDoubleListType:
      return Rcpp::wrap(
          XPtrTorchvector_double(lantern_IValue_DoubleList(self->get())));

    case IValue_types::IValueGeneratorType:
      return Rcpp::wrap(
          XPtrTorchGenerator(lantern_IValue_Generator(self->get())));

    case IValue_types::IValueIntType:
      return Rcpp::wrap(lantern_IValue_Int(self->get()));

    case IValue_types::IValueIntListType:
      return Rcpp::wrap(
          XPtrTorchvector_int64_t(lantern_IValue_IntList(self->get())));

    case IValue_types::IValueModuleType:
      return Rcpp::wrap(
          XPtrTorchScriptModule(lantern_IValue_Module(self->get())));

    case IValue_types::IValueNoneType:
      return R_NilValue;

    case IValue_types::IValueScalarType:
      return Rcpp::wrap(XPtrTorchScalar(lantern_IValue_Scalar(self->get())));

    case IValue_types::IValueStringType:
      return Rcpp::wrap(XPtrTorchstring(lantern_IValue_String(self->get())));

    case IValue_types::IValueTensorListType:
      return Rcpp::wrap(
          XPtrTorchTensorList(lantern_IValue_TensorList(self->get())));

    case IValue_types::IValueTensorType: {
      auto ten = PROTECT(XPtrTorchTensor(lantern_IValue_Tensor(self->get())));
      auto sxp = Rcpp::wrap(ten);
      UNPROTECT(1);
      return sxp;
    }

    case IValue_types::IValueTupleType:
      return Rcpp::wrap(
          XPtrTorchNamedTupleHelper(lantern_IValue_Tuple(self->get())));

    case IValue_types::IValueGenericDictType:
      return Rcpp::wrap(
          XPtrTorchGenericDict(lantern_IValue_GenericDict(self->get())));

    case IValue_types::IValueListType:
      return Rcpp::wrap(XPtrTorchGenericList(lantern_IValue_List(self->get())));
  }

  Rcpp::Rcout << lantern_IValue_type(self->get()) << std::endl;
  Rcpp::stop("Type not handled");
}

XPtrTorchIValue from_sexp_ivalue(SEXP x) {
  if (TYPEOF(x) == INTSXP && LENGTH(x) == 1 && Rf_inherits(x, "jit_scalar")) {
    return XPtrTorchIValue(lantern_IValue_from_Int(Rcpp::as<int64_t>(x)));
  }

  if (TYPEOF(x) == INTSXP) {
    return XPtrTorchIValue(lantern_IValue_from_IntList(
        Rcpp::as<XPtrTorchvector_int64_t>(x).get()));
  }

  if (TYPEOF(x) == LGLSXP && LENGTH(x) == 1 && Rf_inherits(x, "jit_scalar")) {
    return XPtrTorchIValue(lantern_IValue_from_Bool(Rcpp::as<bool>(x)));
  }

  if (TYPEOF(x) == LGLSXP) {
    return XPtrTorchIValue(
        lantern_IValue_from_BoolList(Rcpp::as<XPtrTorchvector_bool>(x).get()));
  }

  if (TYPEOF(x) == REALSXP && LENGTH(x) == 1 && Rf_inherits(x, "jit_scalar")) {
    return XPtrTorchIValue(lantern_IValue_from_Double(Rcpp::as<double>(x)));
  }

  if (TYPEOF(x) == REALSXP) {
    return XPtrTorchIValue(lantern_IValue_from_DoubleList(
        Rcpp::as<XPtrTorchvector_double>(x).get()));
  }

  if (TYPEOF(x) == STRSXP && LENGTH(x) == 1) {
    return XPtrTorchIValue(
        lantern_IValue_from_String(Rcpp::as<XPtrTorchstring>(x).get()));
  }

  if (is_tensor(x)) {
    return XPtrTorchIValue(
        lantern_IValue_from_Tensor(Rcpp::as<XPtrTorchTensor>(x).get()));
  }

  if (x == R_NilValue) {
    return XPtrTorchIValue(lantern_IValue_from_None());
  }

  // is a list
  if (TYPEOF(x) == VECSXP) {
    auto x_ = Rcpp::as<Rcpp::List>(x);

    // is a named list, thus we should convert to a dictionary to
    // preserve the names
    if (rlang_is_named(x)) {
      if (!Rf_inherits(x, "jit_tuple")) {
        // named list of tensors! we will convert to a Dict
        if (std::all_of(x_.cbegin(), x_.cend(),
                        [](SEXP x) { return is_tensor(x); })) {
          return XPtrTorchIValue(lantern_IValue_from_TensorDict(
              Rcpp::as<XPtrTorchTensorDict>(x).get()));
        }
      }

      // named list of arbitrary types. converting to a NamedTuple
      return XPtrTorchIValue(lantern_IValue_from_NamedTuple(
          Rcpp::as<XPtrTorchNamedTupleHelper>(x).get()));
    } else {
      if (!Rf_inherits(x, "jit_tuple")) {
        if (std::all_of(x_.cbegin(), x_.cend(),
                        [](SEXP x) { return is_tensor(x); })) {
          return XPtrTorchIValue(lantern_IValue_from_TensorList(
              Rcpp::as<XPtrTorchTensorList>(x).get()));
        }
      }

      // it's not a list full of tensors so we create a tuple
      return XPtrTorchIValue(
          lantern_IValue_from_Tuple(Rcpp::as<XPtrTorchTuple>(x).get()));
    }
  }

  Rcpp::stop("Unsupported type");
}

void delete_ivalue(void* x) { lantern_IValue_delete(x); }

// tuple

SEXP operator_sexp_tuple(const XPtrTorchTuple* self) {
  auto size = lantern_jit_Tuple_size(self->get());

  Rcpp::List out;
  for (int i = 0; i < size; i++) {
    out.push_back(
        Rcpp::wrap(XPtrTorchIValue(lantern_jit_Tuple_at(self->get(), i))));
  }

  return out;
}

XPtrTorchTuple from_sexp_tuple(SEXP x) {
  auto list = Rcpp::as<Rcpp::List>(x);
  XPtrTorchTuple out = lantern_jit_Tuple_new();
  for (auto el : list) {
    lantern_jit_Tuple_push_back(out.get(), Rcpp::as<XPtrTorchIValue>(el).get());
  }
  return out;
}

void delete_tuple(void* x) { lantern_jit_Tuple_delete(x); }

// named tuple helper

SEXP operator_sexp_named_tuple_helper(const XPtrTorchNamedTupleHelper* self) {
  XPtrTorchvector_string names = lantern_jit_NamedTupleHelper_keys(self->get());
  XPtrTorchvector_IValue elements =
      lantern_jit_NamedTupleHelper_elements(self->get());

  std::vector<std::string> names_ = Rcpp::as<std::vector<std::string>>(names);
  Rcpp::List elements_ = Rcpp::as<Rcpp::List>(elements);

  if (names_.size() != elements_.size()) return elements_;

  if (std::all_of(names_.begin(), names_.end(),
                  [](std::string x) { return x.length() != 0; })) {
    elements_.attr("names") = names_;
  }

  return elements_;
}

XPtrTorchNamedTupleHelper from_sexp_named_tuple_helper(SEXP x) {
  XPtrTorchNamedTupleHelper out = lantern_jit_NamedTuple_new();
  auto x_ = Rcpp::as<Rcpp::List>(x);
  auto names = Rcpp::as<std::vector<std::string>>(x_.names());

  for (int i = 0; i < names.size(); i++) {
    lantern_jit_NamedTuple_push_back(out.get(), XPtrTorchstring(names[i]).get(),
                                     Rcpp::as<XPtrTorchIValue>(x_[i]).get());
  }

  return out;
}

void delete_named_tuple_helper(void* x) { lantern_NamedTupleHelper_delete(x); }

// vector_ivalue

SEXP operator_sexp_vector_ivalue(const XPtrTorchvector_IValue* self) {
  auto size = lantern_jit_vector_IValue_size(self->get());

  Rcpp::List out;
  for (int i = 0; i < size; i++) {
    out.push_back(Rcpp::wrap(
        XPtrTorchIValue(lantern_jit_vector_IValue_at(self->get(), i))));
  }

  return out;
}

void delete_vector_ivalue(void* x) { lantern_jit_vector_IValue_delete(x); }

// generic dict

SEXP operator_sexp_generic_dict(const XPtrTorchGenericDict* self) {
  XPtrTorchvector_IValue keys = lantern_jit_GenericDict_keys(self->get());
  int64_t size = lantern_jit_vector_IValue_size(keys.get());

  Rcpp::List out;
  for (int i = 0; i < size; i++) {
    out.push_back(XPtrTorchIValue(lantern_jit_GenericDict_at(
        self->get(),
        XPtrTorchIValue(lantern_jit_vector_IValue_at(keys.get(), i)).get())));
  }
  out.attr("names") = Rcpp::wrap(keys);
  return out;
}

void delete_generic_dict(void* x) { lantern_jit_GenericDict_delete(x); }

// generic list

SEXP operator_sexp_generic_list(const XPtrTorchGenericList* self) {
  int64_t size = lantern_jit_GenericList_size(self->get());

  Rcpp::List out;
  for (int i = 0; i < size; i++) {
    out.push_back(XPtrTorchIValue(lantern_jit_GenericList_at(self->get(), i)));
  }

  return out;
}

void delete_generic_list(void* x) { lantern_jit_GenericList_delete(x); }

// optional_tensor_list

XPtrTorchOptionalTensorList from_sexp_optional_tensor_list(SEXP x) {
  if (TYPEOF(x) == EXTPTRSXP && Rf_inherits(x, "torch_tensor")) {
    Rcpp::List tmp = Rcpp::List::create(x);
    return cpp_torch_optional_tensor_list(tmp);
  }

  if (Rf_isVectorAtomic(x)) {
    Rcpp::List tmp = Rcpp::List::create(x);
    return cpp_torch_optional_tensor_list(tmp);
  }

  if (TYPEOF(x) == VECSXP) {
    return cpp_torch_optional_tensor_list(Rcpp::as<Rcpp::List>(x));
  }

  Rcpp::stop("Expected a torch_optional_tensor_list.");
}

void delete_optional_tensor_list(void* x) {
  lantern_OptionalTensorList_delete(x);
}

// index_tensor_list

XPtrTorchIndexTensorList from_sexp_index_tensor_list(SEXP x) {
  XPtrTorchTensorList t = from_sexp_tensor_list(x);
  XPtrTorchIndexTensorList zero_index = to_index_tensor_list(t);

  return zero_index;
}

// optional_index_tensor_list

XPtrTorchOptionalIndexTensorList from_sexp_optional_index_tensor_list(SEXP x) {
  XPtrTorchOptionalTensorList t = from_sexp_optional_tensor_list(x);
  XPtrTorchOptionalIndexTensorList zero_index =
      to_optional_index_tensor_list(t);

  return zero_index;
}

// optional device

XPtrTorchOptionalDevice from_sexp_optional_device(SEXP x) {
  if (TYPEOF(x) == NILSXP) {
    return XPtrTorchOptionalDevice(lantern_optional_device(nullptr));
  } else {
    return XPtrTorchOptionalDevice(
        lantern_optional_device(Rcpp::as<XPtrTorchDevice>(x).get()));
  }
}

void delete_optional_device(void* x) { lantern_optional_device_delete(x); }

// int array ref

XPtrTorchIntArrayRef from_sexp_int_array_ref(SEXP x, bool allow_null,
                                             bool index) {
  if (TYPEOF(x) == NILSXP) {
    if (allow_null) {
      return nullptr;
    } else {
      Rcpp::stop("Expected a list of integers and found NULL.");
    }
  }

  std::vector<int64_t> vec;

  if (TYPEOF(x) == VECSXP) {
    auto tmp = Rcpp::as<Rcpp::List>(x);
    for (auto i = tmp.begin(); i != tmp.end(); ++i) {
      vec.push_back(Rcpp::as<int64_t>(*i));
    }
  } else {
    vec = Rcpp::as<std::vector<int64_t>>(x);
  }

  if (index) {
    for (int i = 0; i < vec.size(); i++) {
      if (vec[i] == 0) {
        Rcpp::stop("Indexing starts at 1 but found a 0.");
      }

      if (vec[i] > 0) {
        vec[i] = vec[i] - 1;
      }
    }
  }

  auto ptr = lantern_vector_int64_t(vec.data(), vec.size());
  return XPtrTorchIntArrayRef(ptr);
}

// optional double array ref

XPtrTorchOptionalDoubleArrayRef from_sexp_optional_double_array_ref(SEXP x) {
  if (TYPEOF(x) == NILSXP || LENGTH(x) == 0) {
    return XPtrTorchOptionalDoubleArrayRef(
        lantern_optional_vector_double(NULL, 0, true));
  }

  // handle lists of double
  std::vector<double> data;
  if (TYPEOF(x) == VECSXP) {
    auto tmp = Rcpp::as<Rcpp::List>(x);
    for (auto i = tmp.begin(); i != tmp.end(); ++i) {
      data.push_back(Rcpp::as<double>(*i));
    }
  } else {
    data = Rcpp::as<std::vector<double>>(x);
  }

  return XPtrTorchOptionalDoubleArrayRef(
      lantern_optional_vector_double(data.data(), data.size(), false));
}

void delete_optional_double_array_ref(void* x) {
  lantern_optional_vector_double_delete(x);
}

// optional int array ref

XPtrTorchOptionalIntArrayRef from_sexp_optional_int_array_ref(SEXP x,
                                                              bool index) {
  bool is_null;
  std::vector<int64_t> data;

  if (TYPEOF(x) == NILSXP || LENGTH(x) == 0) {
    is_null = true;
  } else {
    // handle lists of integers
    if (TYPEOF(x) == VECSXP) {
      auto tmp = Rcpp::as<Rcpp::List>(x);
      for (auto i = tmp.begin(); i != tmp.end(); ++i) {
        data.push_back(Rcpp::as<int64_t>(*i));
      }
    } else {
      data = Rcpp::as<std::vector<int64_t>>(x);
    }

    if (index) {
      for (int i = 0; i < data.size(); i++) {
        if (data[i] == 0) {
          Rcpp::stop("Indexing starts at 1 but found a 0.");
        }

        if (data[i] > 0) {
          data[i] = data[i] - 1;
        }
      }
    }

    is_null = false;
  }

  return XPtrTorchOptionalIntArrayRef(
      lantern_optional_vector_int64_t(data.data(), data.size(), is_null));
}

void delete_optional_int_array_ref(void* x) {
  lantern_optional_vector_int64_t_delete(x);
}

// tensor_dict

XPtrTorchTensorDict from_sexp_tensor_dict(SEXP x) {
  auto list = Rcpp::as<Rcpp::List>(x);
  XPtrTorchTensorDict out = lantern_jit_TensorDict_new();

  Rcpp::List names = list.attr("names");
  for (int i = 0; i < names.size(); i++) {
    lantern_jit_TensorDict_push_back(out.get(),
                                     Rcpp::as<XPtrTorchstring>(names[i]).get(),
                                     Rcpp::as<XPtrTorchTensor>(list[i]).get());
  }
  return out;
}

void delete_tensor_dict(void* x) { lantern_jit_TensorDict_delete(x); }

// optional_int64_t

XPtrTorchoptional_int64_t from_sexp_optional_int64_t(SEXP x) {
  if (TYPEOF(x) == NILSXP || LENGTH(x) == 0) {
    return XPtrTorchoptional_int64_t(lantern_optional_int64_t(nullptr));
  }

  return XPtrTorchoptional_int64_t(
      lantern_optional_int64_t(Rcpp::as<XPtrTorchint64_t>(x).get()));
}

SEXP operator_sexp_optional_int64_t(const XPtrTorchoptional_int64_t* x) {
  if (!lantern_optional_int64_t_has_value(x->get())) {
    return R_NilValue;
  }

  return Rcpp::wrap(XPtrTorchint64_t(lantern_optional_int64_t_value(x->get())));
}

void delete_optional_int64_t(void* x) { lantern_optional_int64_t_delete(x); }

// index_int64_t

XPtrTorchindex_int64_t from_sexp_index_int64_t(SEXP x_) {
  int64_t x = 0;

  if (LENGTH(x_) > 0) {
    x = Rcpp::as<int64_t>(x_);
  }

  if (x == 0) {
    Rcpp::stop("Indexing starts at 1 but found a 0.");
  }

  if (x > 0) {
    x = x - 1;
  }

  return XPtrTorchindex_int64_t(
      std::shared_ptr<void>(lantern_int64_t(x), lantern_int64_t_delete));
}

// optional index int64_t

XPtrTorchoptional_index_int64_t from_sexp_optional_index_int64_t(SEXP x_) {
  if (TYPEOF(x_) == NILSXP || LENGTH(x_) == 0) {
    return XPtrTorchoptional_index_int64_t(
        Rcpp::as<XPtrTorchoptional_int64_t>(x_).get_shared());
  }

  auto x = Rcpp::as<int64_t>(x_);
  if (x == 0) {
    Rcpp::stop("Indexing starts at 1 but found a 0.");
  }

  if (x > 0) {
    x = x - 1;
  }

  return XPtrTorchoptional_index_int64_t(
      Rcpp::as<XPtrTorchoptional_int64_t>(Rcpp::wrap(x)).get_shared());
}

// function ptr

void delete_function_ptr(void* x) { lantern_FunctionPtr_delete(x); }

// QScheme

void delete_qscheme(void* x) { lantern_QScheme_delete(x); }

// double

SEXP operator_sexp_double(const XPtrTorchdouble* x) {
  return Rcpp::wrap(lantern_double_get(x->get()));
}

XPtrTorchdouble from_sexp_double(SEXP x) {
  return XPtrTorchdouble(lantern_double(Rcpp::as<double>(x)));
}

void delete_double(void* x) { lantern_double_delete(x); }

// optional double

SEXP operator_sexp_optional_double(const XPtrTorchOptionaldouble* x) {
  if (!lantern_optional_double_has_value(x->get())) {
    return R_NilValue;
  }

  return Rcpp::wrap(XPtrTorchdouble(lantern_optional_double_value(x->get())));
}

XPtrTorchOptionaldouble from_sexp_optional_double(SEXP x) {
  if (TYPEOF(x) == NILSXP || LENGTH(x) == 0) {
    return XPtrTorchOptionaldouble(lantern_optional_double(nullptr));
  }

  return XPtrTorchOptionaldouble(
      lantern_optional_double(Rcpp::as<XPtrTorchdouble>(x).get()));
}

void delete_optional_double(void* x) { lantern_optional_double_delete(x); }

// variable_list

XPtrTorchvariable_list from_sexp_variable_list(SEXP x) {
  auto list = Rcpp::as<Rcpp::List>(x);
  XPtrTorchvariable_list out = lantern_variable_list_new();

  for (int i = 0; i < list.length(); i++) {
    lantern_variable_list_push_back(out.get(),
                                    Rcpp::as<XPtrTorchTensor>(list[i]).get());
  }

  return out;
}

SEXP operator_sexp_variable_list(const XPtrTorchvariable_list* self) {
  Rcpp::List out;
  int64_t sze = lantern_variable_list_size(self->get());

  for (int64_t i = 0; i < sze; i++) {
    out.push_back(XPtrTorchTensor(lantern_variable_list_get(self->get(), i)));
  }

  return out;
}

void delete_variable_list(void* x) { lantern_variable_list_delete(x); }

// int64_t

XPtrTorchint64_t from_sexp_int64_t(SEXP x) {
  return XPtrTorchint64_t(lantern_int64_t(Rcpp::as<int64_t>(x)));
}

SEXP operator_sexp_int64_t(const XPtrTorchint64_t* x) {
  return Rcpp::wrap(lantern_int64_t_get(x->get()));
}

void delete_int64_t(void* x) { lantern_int64_t_delete(x); }

// bool

XPtrTorchbool from_sexp_bool(SEXP x) {
  return XPtrTorchbool(lantern_bool(Rcpp::as<bool>(x)));
}

SEXP operator_sexp_bool(const XPtrTorchbool* x) {
  return Rcpp::wrap(lantern_bool_get(x->get()));
}

void delete_bool(void* x) { lantern_bool_delete(x); }

// optional bool

XPtrTorchoptional_bool from_sexp_optional_bool(SEXP x) {
  if (TYPEOF(x) == NILSXP) {
    return XPtrTorchoptional_bool(lantern_optional_bool(nullptr));
  }

  return XPtrTorchoptional_bool(
      lantern_optional_bool(Rcpp::as<XPtrTorchbool>(x).get()));
}

SEXP operator_sexp_optional_bool(const XPtrTorchoptional_bool* x) {
  if (!lantern_optional_bool_has_value(x->get())) {
    return R_NilValue;
  }

  return Rcpp::wrap(XPtrTorchbool(lantern_optional_bool_value(x->get())));
}

void delete_optional_bool(void* x) { lantern_optional_bool_delete(x); }

// layout

void delete_layout(void* x) { lantern_Layout_delete(x); }

// tensor index

void delete_tensor_index(void* x) { lantern_TensorIndex_delete(x); }

// slice

void delete_slice(void* x) { lantern_Slice_delete(x); }

// packed_sequence

void delete_packed_sequence(void* x) { lantern_PackedSequence_delete(x); }

// storage

void delete_storage(void* x) { lantern_Storage_delete(x); }

// jit module

void delete_jit_module(void* x) { lantern_JITModule_delete(x); }

// traceable_function

void delete_traceable_function(void* x) { lantern_TraceableFunction_delete(x); }

// vector_void

void delete_vector_void(void* x) { lantern_vector_void_delete(x); }
