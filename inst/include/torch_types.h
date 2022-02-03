#ifndef TORCH_TYPES
#define TORCH_TYPES

#include <Rcpp.h>

#include <memory>

#include "torch_deleters.h"

class XPtrTorch {
 private:
  std::shared_ptr<void> ptr;

 public:
  XPtrTorch(
      void* x, std::function<void(void*)> deleter = [](void*) {})
      : XPtrTorch(std::shared_ptr<void>(x, deleter)){};
  explicit XPtrTorch(std::shared_ptr<void> x) : ptr(x) {}
  void* get() const { return ptr.get(); }
  std::shared_ptr<void> get_shared() const { return ptr; }
};

class XPtrTorchIndexTensor : public XPtrTorch {
 public:
  XPtrTorchIndexTensor() : XPtrTorch{NULL} {}
  XPtrTorchIndexTensor(void* x) : XPtrTorch(x, delete_tensor) {}
  explicit XPtrTorchIndexTensor(std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchIndexTensor(const XPtrTorchIndexTensor& x)
      : XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchIndexTensor(SEXP x);
  operator SEXP() const;
};

class XPtrTorchTensor : public XPtrTorch {
 public:
  // TODO: we should make this explicit at some point, but not currently
  // possible because we rely on it in too many places.
  XPtrTorchTensor() : XPtrTorch{NULL} {}
  XPtrTorchTensor(void* x) : XPtrTorch(x, delete_tensor) {}
  explicit XPtrTorchTensor(std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchTensor(const XPtrTorchTensor& x) : XPtrTorch(x.get_shared()) {}
  XPtrTorchTensor(XPtrTorchIndexTensor x) : XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchTensor(SEXP x);
  operator SEXP() const;
};

class XPtrTorchScriptModule : public XPtrTorch {
 public:
  XPtrTorchScriptModule(void* x) : XPtrTorch(x, delete_script_module) {}
  explicit XPtrTorchScriptModule(std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchScriptModule(const XPtrTorchScriptModule& x)
      : XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchScriptModule(SEXP x);
  operator SEXP() const;
};

class XPtrTorchScriptMethod : public XPtrTorch {
 public:
  XPtrTorchScriptMethod(void* x) : XPtrTorch(x, delete_script_method) {}
  XPtrTorchScriptMethod(void* x, std::function<void(void* x)> deleter)
      : XPtrTorch(x, deleter) {}
  explicit XPtrTorchScriptMethod(std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchScriptMethod(const XPtrTorchScriptMethod& x)
      : XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchScriptMethod(SEXP x);
  operator SEXP() const;
};

class XPtrTorchOptionalTensor : public XPtrTorch {
 public:
  XPtrTorchOptionalTensor(void* x) : XPtrTorch(x, delete_optional_tensor) {}
  explicit XPtrTorchOptionalTensor(std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchOptionalTensor(const XPtrTorchOptionalTensor& x)
      : XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchOptionalTensor(SEXP x);
  operator SEXP() const;
};

class XPtrTorchTensorList : public XPtrTorch {
 public:
  XPtrTorchTensorList(void* x) : XPtrTorch(x, delete_tensor_list) {}
  explicit XPtrTorchTensorList(std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchTensorList(const XPtrTorchTensorList& x)
      : XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchTensorList(SEXP x);
  operator SEXP() const;
};

class XPtrTorchOptionalTensorList : public XPtrTorch {
 public:
  XPtrTorchOptionalTensorList(void* x)
      : XPtrTorch(x, delete_optional_tensor_list) {}
  explicit XPtrTorchOptionalTensorList(std::shared_ptr<void> x)
      : XPtrTorch(x) {}
  XPtrTorchOptionalTensorList(const XPtrTorchTensorList& x)
      : XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchOptionalTensorList(SEXP x);
  operator SEXP() const;
};

class XPtrTorchIndexTensorList : public XPtrTorch {
 public:
  XPtrTorchIndexTensorList(void* x) : XPtrTorch(x, delete_tensor_list) {}
  explicit XPtrTorchIndexTensorList(std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchIndexTensorList(const XPtrTorchIndexTensorList& x)
      : XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchIndexTensorList(SEXP x);
};

class XPtrTorchOptionalIndexTensorList : public XPtrTorch {
 public:
  XPtrTorchOptionalIndexTensorList(void* x)
      : XPtrTorch(x, delete_optional_tensor_list) {}
  explicit XPtrTorchOptionalIndexTensorList(std::shared_ptr<void> x)
      : XPtrTorch(x) {}
  XPtrTorchOptionalIndexTensorList(const XPtrTorchOptionalIndexTensorList& x)
      : XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchOptionalIndexTensorList(SEXP x);
};

class XPtrTorchScalarType : public XPtrTorch {
 public:
  XPtrTorchScalarType(void* x) : XPtrTorch(x, delete_scalar_type) {}
  explicit XPtrTorchScalarType(std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchScalarType(const XPtrTorchScalarType& x)
      : XPtrTorch(x.get_shared()) {}
  XPtrTorchScalarType(SEXP x);
  operator SEXP() const;
};

class XPtrTorchoptional_scalar_type : public XPtrTorch {
 public:
  XPtrTorchoptional_scalar_type(void* x)
      : XPtrTorch(x, delete_optional_scalar_type) {}
  XPtrTorchoptional_scalar_type(SEXP x);
  operator SEXP() const;
};

class XPtrTorchScalar : public XPtrTorch {
 public:
  XPtrTorchScalar() : XPtrTorch{NULL} {}
  XPtrTorchScalar(void* x) : XPtrTorch(x, delete_scalar) {}
  explicit XPtrTorchScalar(std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchScalar(const XPtrTorchScalar& x) : XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchScalar(SEXP x);
  operator SEXP() const;
};

class XPtrTorchTensorOptions : public XPtrTorch {
 public:
  XPtrTorchTensorOptions(void* x) : XPtrTorch(x, delete_tensor_options){};
  explicit XPtrTorchTensorOptions(std::shared_ptr<void> x) : XPtrTorch(x){};
  XPtrTorchTensorOptions(const XPtrTorchTensorOptions& x)
      : XPtrTorch(x.get_shared()){};
  explicit XPtrTorchTensorOptions(SEXP x);
  operator SEXP() const;
};

class XPtrTorchDevice : public XPtrTorch {
 public:
  XPtrTorchDevice(void* x) : XPtrTorch(x, delete_device) {}
  explicit XPtrTorchDevice(std::shared_ptr<void> x) : XPtrTorch(x){};
  XPtrTorchDevice(const XPtrTorchDevice& x) : XPtrTorch(x.get_shared()){};
  explicit XPtrTorchDevice(SEXP x);
  operator SEXP() const;
};

class XPtrTorchOptionalDevice : public XPtrTorch {
 public:
  XPtrTorchOptionalDevice(void* x) : XPtrTorch(x, delete_optional_device) {}
  explicit XPtrTorchOptionalDevice(std::shared_ptr<void> x) : XPtrTorch(x){};
  XPtrTorchOptionalDevice(const XPtrTorchOptionalDevice& x)
      : XPtrTorch(x.get_shared()){};
  explicit XPtrTorchOptionalDevice(SEXP x);
  operator SEXP() const;
};

class XPtrTorchDtype : public XPtrTorch {
 public:
  XPtrTorchDtype() : XPtrTorch{NULL} {}
  XPtrTorchDtype(void* x) : XPtrTorch(x, delete_dtype) {}
  explicit XPtrTorchDtype(std::shared_ptr<void> x) : XPtrTorch(x){};
  XPtrTorchDtype(const XPtrTorchDtype& x) : XPtrTorch(x.get_shared()){};
  explicit XPtrTorchDtype(SEXP x);
  operator SEXP() const;
};

class XPtrTorchDimname : public XPtrTorch {
 public:
  XPtrTorchDimname(void* x) : XPtrTorch(x, delete_dimname) {}
  explicit XPtrTorchDimname(std::shared_ptr<void> x) : XPtrTorch(x){};
  XPtrTorchDimname(const XPtrTorchDimname& x) : XPtrTorch(x.get_shared()){};
  explicit XPtrTorchDimname(SEXP x);
  explicit XPtrTorchDimname(const std::string& x)
      : XPtrTorchDimname(Rcpp::wrap(x)){};
  operator SEXP() const;
};

class XPtrTorchDimnameList : public XPtrTorch {
 public:
  XPtrTorchDimnameList(void* x) : XPtrTorch(x, delete_dimname_list) {}
  explicit XPtrTorchDimnameList(std::shared_ptr<void> x) : XPtrTorch(x){};
  XPtrTorchDimnameList(const XPtrTorchDimnameList& x)
      : XPtrTorch(x.get_shared()){};
  explicit XPtrTorchDimnameList(SEXP x);
  operator SEXP() const;
};

class XPtrTorchOptionalDimnameList : public XPtrTorch {
 public:
  XPtrTorchOptionalDimnameList(void* x)
      : XPtrTorch(x, delete_optional_dimname_list) {}
  explicit XPtrTorchOptionalDimnameList(std::shared_ptr<void> x)
      : XPtrTorch(x){};
  XPtrTorchOptionalDimnameList(const XPtrTorchOptionalDimnameList& x)
      : XPtrTorch(x.get_shared()){};
  explicit XPtrTorchOptionalDimnameList(SEXP x);
  operator SEXP() const;
};

class XPtrTorchjit_named_parameter_list : public XPtrTorch {
 public:
  XPtrTorchjit_named_parameter_list(void* x)
      : XPtrTorch(x, delete_jit_named_parameter_list) {}
  explicit XPtrTorchjit_named_parameter_list(std::shared_ptr<void> x)
      : XPtrTorch(x){};
  XPtrTorchjit_named_parameter_list(const XPtrTorchjit_named_parameter_list& x)
      : XPtrTorch(x.get_shared()){};
  operator SEXP() const;
};

class XPtrTorchjit_named_buffer_list : public XPtrTorch {
 public:
  XPtrTorchjit_named_buffer_list(void* x)
      : XPtrTorch(x, delete_jit_named_buffer_list) {}
  explicit XPtrTorchjit_named_buffer_list(std::shared_ptr<void> x)
      : XPtrTorch(x){};
  XPtrTorchjit_named_buffer_list(const XPtrTorchjit_named_buffer_list& x)
      : XPtrTorch(x.get_shared()){};
  operator SEXP() const;
};

class XPtrTorchjit_named_module_list : public XPtrTorch {
 public:
  XPtrTorchjit_named_module_list(void* x)
      : XPtrTorch(x, delete_jit_named_module_list) {}
  explicit XPtrTorchjit_named_module_list(std::shared_ptr<void> x)
      : XPtrTorch(x){};
  XPtrTorchjit_named_module_list(const XPtrTorchjit_named_module_list& x)
      : XPtrTorch(x.get_shared()){};
  operator SEXP() const;
};

class XPtrTorchGenerator : public XPtrTorch {
 public:
  XPtrTorchGenerator(void* x) : XPtrTorch(x, delete_generator) {}
  explicit XPtrTorchGenerator(std::shared_ptr<void> x) : XPtrTorch(x){};
  XPtrTorchGenerator(const XPtrTorchGenerator& x) : XPtrTorch(x.get_shared()){};
  explicit XPtrTorchGenerator(SEXP x);
  operator SEXP() const;
};

class XPtrTorchOptionalGenerator : public XPtrTorch {
 public:
  XPtrTorchOptionalGenerator(void* x)
      : XPtrTorch(x, delete_optional_generator) {}
  explicit XPtrTorchOptionalGenerator(std::shared_ptr<void> x) : XPtrTorch(x){};
  XPtrTorchOptionalGenerator(const XPtrTorchOptionalGenerator& x)
      : XPtrTorch(x.get_shared()){};
  explicit XPtrTorchOptionalGenerator(SEXP x);
  operator SEXP() const;
};

class XPtrTorchMemoryFormat : public XPtrTorch {
 public:
  XPtrTorchMemoryFormat() : XPtrTorch{NULL} {};
  XPtrTorchMemoryFormat(void* x) : XPtrTorch(x, delete_memory_format) {}
  explicit XPtrTorchMemoryFormat(std::shared_ptr<void> x) : XPtrTorch(x){};
  XPtrTorchMemoryFormat(const XPtrTorchMemoryFormat& x)
      : XPtrTorch(x.get_shared()){};
  explicit XPtrTorchMemoryFormat(SEXP x);
  operator SEXP() const;
};

class XPtrTorchIntArrayRef : public XPtrTorch {
 public:
  XPtrTorchIntArrayRef() : XPtrTorch{NULL} {};
  XPtrTorchIntArrayRef(void* x) : XPtrTorch(x, delete_vector_int64_t) {}
  explicit XPtrTorchIntArrayRef(std::shared_ptr<void> x) : XPtrTorch(x){};
  XPtrTorchIntArrayRef(const XPtrTorchIntArrayRef& x)
      : XPtrTorch(x.get_shared()){};
  explicit XPtrTorchIntArrayRef(SEXP x);
  // operator SEXP () const;
};

class XPtrTorchFunctionPtr : public XPtrTorch {
 public:
  XPtrTorchFunctionPtr(void* x) : XPtrTorch(x, delete_function_ptr) {}
  XPtrTorchFunctionPtr(void* x, std::function<void(void*)> deleter)
      : XPtrTorch(x, deleter) {}
  explicit XPtrTorchFunctionPtr(std::shared_ptr<void> x) : XPtrTorch(x){};
  XPtrTorchFunctionPtr(const XPtrTorchFunctionPtr& x)
      : XPtrTorch(x.get_shared()){};
};

class XPtrTorchIndexIntArrayRef : public XPtrTorch {
 public:
  XPtrTorchIndexIntArrayRef() : XPtrTorch{NULL} {};
  XPtrTorchIndexIntArrayRef(void* x) : XPtrTorch(x, delete_vector_int64_t) {}
  explicit XPtrTorchIndexIntArrayRef(std::shared_ptr<void> x) : XPtrTorch(x){};
  XPtrTorchIndexIntArrayRef(const XPtrTorchIndexIntArrayRef& x)
      : XPtrTorch(x.get_shared()){};
  explicit XPtrTorchIndexIntArrayRef(SEXP x);
};

class XPtrTorchOptionalDoubleArrayRef : public XPtrTorch {
 public:
  XPtrTorchOptionalDoubleArrayRef() : XPtrTorch{NULL} {};
  XPtrTorchOptionalDoubleArrayRef(void* x)
      : XPtrTorch(x, delete_optional_double_array_ref){};
  XPtrTorchOptionalDoubleArrayRef(SEXP x);
};

class XPtrTorchOptionalIntArrayRef : public XPtrTorch {
 public:
  XPtrTorchOptionalIntArrayRef() : XPtrTorch{NULL} {};
  XPtrTorchOptionalIntArrayRef(SEXP x);
  XPtrTorchOptionalIntArrayRef(void* x)
      : XPtrTorch(x, delete_optional_int_array_ref){};
};

class XPtrTorchOptionalIndexIntArrayRef : public XPtrTorchOptionalIntArrayRef {
 public:
  explicit XPtrTorchOptionalIndexIntArrayRef(SEXP x);
};

class XPtrTorchbool : public XPtrTorch {
 public:
  operator SEXP() const;
  XPtrTorchbool(SEXP x_);
  XPtrTorchbool(void* x) : XPtrTorch(x, delete_bool) {}
};

class XPtrTorchoptional_bool : public XPtrTorch {
 public:
  XPtrTorchoptional_bool(SEXP x_);
  XPtrTorchoptional_bool(void* x) : XPtrTorch(x, delete_optional_bool) {}
  operator SEXP() const;
};

class XPtrTorchoptional_memory_format : public XPtrTorch {
 public:
  XPtrTorchoptional_memory_format(SEXP x_);
  XPtrTorchoptional_memory_format(void* x)
      : XPtrTorch(x, delete_optional_memory_format) {}
  operator SEXP() const;
};

class XPtrTorchoptional_scalar : public XPtrTorch {
 public:
  XPtrTorchoptional_scalar(SEXP x_);
  XPtrTorchoptional_scalar(void* x) : XPtrTorch(x, delete_optional_scalar) {}
  operator SEXP() const;
};

class XPtrTorchindex_int64_t {
 public:
  std::shared_ptr<void> ptr;
  explicit XPtrTorchindex_int64_t(SEXP x_);
  explicit XPtrTorchindex_int64_t(std::shared_ptr<void> x) : ptr(x){};
  void* get() { return ptr.get(); }
};

class XPtrTorchoptional_index_int64_t {
 public:
  std::shared_ptr<void> ptr;
  explicit XPtrTorchoptional_index_int64_t(SEXP x_);
  explicit XPtrTorchoptional_index_int64_t(std::shared_ptr<void> x) : ptr(x){};
  void* get() { return ptr.get(); }
};

class XPtrTorchvector_string : public XPtrTorch {
 public:
  XPtrTorchvector_string(SEXP x);
  XPtrTorchvector_string(const XPtrTorchvector_string& x)
      : XPtrTorch(x.get_shared()){};
  XPtrTorchvector_string(void* x) : XPtrTorch(x, delete_vector_string) {}
  operator SEXP() const;
};

class XPtrTorchstring : public XPtrTorch {
 public:
  XPtrTorchstring(void* x) : XPtrTorch(x, delete_string) {}
  XPtrTorchstring(SEXP x);
  XPtrTorchstring(const XPtrTorchstring& x) : XPtrTorch(x.get_shared()){};
  XPtrTorchstring(std::string x)
      : XPtrTorchstring(fixme_new_string(x.c_str())){};
  operator SEXP() const;
};

class XPtrTorchstring_view : public XPtrTorch {
 public:
  XPtrTorchstring_view(void* x) : XPtrTorch(x, delete_string_view) {}
  XPtrTorchstring_view(SEXP x);
  XPtrTorchstring_view(const XPtrTorchstring_view& x)
      : XPtrTorch(x.get_shared()){};
};

class XPtrTorchoptional_string_view : public XPtrTorch {
 public:
  XPtrTorchoptional_string_view(void* x)
      : XPtrTorch(x, delete_optional_string_view) {}
  XPtrTorchoptional_string_view(SEXP x);
  XPtrTorchoptional_string_view(const XPtrTorchstring_view& x)
      : XPtrTorch(x.get_shared()){};
};

class XPtrTorchoptional_string : public XPtrTorch {
 public:
  XPtrTorchoptional_string(void* x) : XPtrTorch(x, delete_optional_string) {}
  XPtrTorchoptional_string(SEXP x);
  operator SEXP() const;
};

class XPtrTorchStack : public XPtrTorch {
 public:
  XPtrTorchStack(void* x) : XPtrTorch(x, delete_stack) {}
  XPtrTorchStack(void* x, std::function<void(void*)> deleter)
      : XPtrTorch(x, deleter) {}
  explicit XPtrTorchStack(std::shared_ptr<void> x) : XPtrTorch(x){};
  XPtrTorchStack(SEXP x);
  operator SEXP() const;
};

class XPtrTorchIValue : public XPtrTorch {
 public:
  XPtrTorchIValue(void* x) : XPtrTorch(x, delete_ivalue) {}
  explicit XPtrTorchIValue(std::shared_ptr<void> x) : XPtrTorch(x){};
  XPtrTorchIValue(const XPtrTorchIValue& x) : XPtrTorch(x.get_shared()){};
  XPtrTorchIValue(SEXP x);
  operator SEXP() const;
};

class XPtrTorchTuple : public XPtrTorch {
 public:
  XPtrTorchTuple(void* x) : XPtrTorch(x, delete_tuple) {}
  XPtrTorchTuple(const XPtrTorchTuple& x) : XPtrTorch(x.get_shared()){};
  XPtrTorchTuple(SEXP x);
  operator SEXP() const;
};

class XPtrTorchvector_bool : public XPtrTorch {
 public:
  XPtrTorchvector_bool(void* x) : XPtrTorch(x, delete_vector_bool) {}
  operator SEXP() const;
  XPtrTorchvector_bool(SEXP x);
};

class XPtrTorchvector_Scalar : public XPtrTorch {
 public:
  XPtrTorchvector_Scalar(void* x) : XPtrTorch(x, delete_vector_scalar) {}
  operator SEXP() const;
  XPtrTorchvector_Scalar(SEXP x);
  XPtrTorchvector_Scalar(const XPtrTorchvector_Scalar& x)
      : XPtrTorch(x.get_shared()){};
};

class XPtrTorchvector_int64_t : public XPtrTorch {
 public:
  XPtrTorchvector_int64_t(void* x) : XPtrTorch(x, delete_vector_int64_t) {}
  operator SEXP() const;
  XPtrTorchvector_int64_t(SEXP x);
};

class XPtrTorchvector_double : public XPtrTorch {
 public:
  XPtrTorchvector_double(void* x) : XPtrTorch(x, delete_vector_double) {}
  operator SEXP() const;
  XPtrTorchvector_double(SEXP x);
};

class XPtrTorchTensorDict : public XPtrTorch {
 public:
  XPtrTorchTensorDict(void* x) : XPtrTorch(x, delete_tensor_dict) {}
  XPtrTorchTensorDict(SEXP x);
};

class XPtrTorchGenericDict : public XPtrTorch {
 public:
  XPtrTorchGenericDict(void* x) : XPtrTorch(x, delete_generic_dict) {}
  operator SEXP() const;
};

class XPtrTorchGenericList : public XPtrTorch {
 public:
  XPtrTorchGenericList(void* x) : XPtrTorch(x, delete_generic_list) {}
  operator SEXP() const;
};

class XPtrTorchvector_IValue : public XPtrTorch {
 public:
  XPtrTorchvector_IValue(void* x) : XPtrTorch(x, delete_vector_ivalue) {}
  operator SEXP() const;
};

class XPtrTorchNamedTupleHelper : public XPtrTorch {
 public:
  XPtrTorchNamedTupleHelper(void* x)
      : XPtrTorch(x, delete_named_tuple_helper) {}
  XPtrTorchNamedTupleHelper(SEXP x);
  operator SEXP() const;
};

class XPtrTorchCompilationUnit : public XPtrTorch {
 public:
  XPtrTorchCompilationUnit(void* x) : XPtrTorch(x, delete_compilation_unit) {}
  XPtrTorchCompilationUnit(SEXP x);
  explicit XPtrTorchCompilationUnit(std::shared_ptr<void> x) : XPtrTorch(x){};
  XPtrTorchCompilationUnit(const XPtrTorchCompilationUnit& x)
      : XPtrTorch(x.get_shared()){};
  operator SEXP() const;
};

class XPtrTorchoptional_int64_t : public XPtrTorch {
 public:
  XPtrTorchoptional_int64_t(void* x) : XPtrTorch(x, delete_optional_int64_t) {}
  operator SEXP() const;
  XPtrTorchoptional_int64_t(SEXP x);
};

class XPtrTorchdouble : public XPtrTorch {
 public:
  operator SEXP() const;
  XPtrTorchdouble(void* x) : XPtrTorch(x, delete_double) {}
  XPtrTorchdouble(SEXP x);
};

class XPtrTorchOptionaldouble : public XPtrTorch {
 public:
  operator SEXP() const;
  XPtrTorchOptionaldouble(void* x) : XPtrTorch(x, delete_optional_double) {}
  XPtrTorchOptionaldouble(SEXP x);
};

#include <Rcpp.h>

class XPtrTorchQScheme : public XPtrTorch {
 public:
  XPtrTorchQScheme(void* x) : XPtrTorch(x, delete_qscheme) {}
};

class XPtrTorchvariable_list : public XPtrTorch {
 public:
  XPtrTorchvariable_list(void* x) : XPtrTorch(x, delete_variable_list) {}
  XPtrTorchvariable_list(SEXP x);
  operator SEXP() const;
};

class XPtrTorchint64_t : public XPtrTorch {
 public:
  operator SEXP() const;
  XPtrTorchint64_t(void* x) : XPtrTorch(x, delete_int64_t) {}
  XPtrTorchint64_t(SEXP x);
};

class XPtrTorchLayout : public XPtrTorch {
 public:
  XPtrTorchLayout(void* x) : XPtrTorch(x, delete_layout) {}
};

class XPtrTorchTensorIndex : public XPtrTorch {
 public:
  XPtrTorchTensorIndex(void* x) : XPtrTorch(x, delete_tensor_index) {}
};

class XPtrTorchSlice : public XPtrTorch {
 public:
  XPtrTorchSlice(void* x) : XPtrTorch(x, delete_slice) {}
};

class XPtrTorchPackedSequence : public XPtrTorch {
 public:
  XPtrTorchPackedSequence(void* x) : XPtrTorch(x, delete_packed_sequence) {}
};

class XPtrTorchStorage : public XPtrTorch {
 public:
  XPtrTorchStorage(void* x) : XPtrTorch(x, delete_storage) {}
};

class XPtrTorchJITModule : public XPtrTorch {
 public:
  XPtrTorchJITModule(void* x) : XPtrTorch(x, delete_jit_module) {}
};

class XPtrTorchTraceableFunction : public XPtrTorch {
 public:
  XPtrTorchTraceableFunction(void* x)
      : XPtrTorch(x, delete_traceable_function) {}
};

class XPtrTorchvector_void : public XPtrTorch {
 public:
  XPtrTorchvector_void(void* x) : XPtrTorch(x, delete_vector_void) {}
};

template <class T>
class nullable {
 public:
  T x;
  bool is_null = false;
  nullable(Rcpp::Nullable<T> x) {
    if (x.isNotNull()) {
      this->x = Rcpp::as<T>(x);
    } else {
      this->is_null = true;
    }
  };
  void* get() {
    if (this->is_null)
      return nullptr;
    else
      return &(this->x);
  }
  T get_value() {
    if (this->is_null)
      return NULL;
    else
      return this->x;
  }
};

template <class T>
class nullableVector {
 public:
  T x = {0};
  bool is_null = false;
  nullableVector(Rcpp::Nullable<T> x) {
    if (x.isNotNull()) {
      this->x = Rcpp::as<T>(x);
    } else {
      this->is_null = true;
    }
  };
  void* get() {
    if (this->is_null)
      return nullptr;
    else
      return &(this->x);
  }
  T get_value() {
    if (this->is_null)
      return NULL;
    else
      return this->x;
  }
};

namespace torch {
using Tensor = XPtrTorchTensor;
using TensorList = XPtrTorchTensorList;
using ScalarType = XPtrTorchScalarType;
using Scalar = XPtrTorchScalar;
using TensorOptions = XPtrTorchTensorOptions;
using Device = XPtrTorchDevice;
using Dtype = XPtrTorchDtype;
using Dimname = XPtrTorchDimname;
using DimnameList = XPtrTorchDimnameList;
using Generator = XPtrTorchGenerator;
using MemoryFormat = XPtrTorchMemoryFormat;
using IntArrayRef = XPtrTorchIntArrayRef;
using TensorDict = XPtrTorchTensorDict;
using CompilationUnit = XPtrTorchCompilationUnit;
using QScheme = XPtrTorchQScheme;
using variable_list = XPtrTorchvariable_list;
using Layout = XPtrTorchLayout;
using Storage = XPtrTorchStorage;

using string = XPtrTorchstring;
using double_t = XPtrTorchdouble;
using int64_t = XPtrTorchint64_t;
using bool_t = XPtrTorchbool;

namespace indexing {
using TensorIndex = XPtrTorchTensorIndex;
using Slice = XPtrTorchSlice;
}  // namespace indexing

namespace impl {
using GenericDict = XPtrTorchGenericDict;
using GenericList = XPtrTorchGenericList;
using NamedTupleHelper = XPtrTorchNamedTupleHelper;
}  // namespace impl

namespace nn {
namespace utils {
namespace rnn {
using PackedSequence = XPtrTorchPackedSequence;
}
}  // namespace utils
}  // namespace nn

namespace optional {
using Tensor = XPtrTorchOptionalTensor;
using TensorList = XPtrTorchOptionalTensorList;
using Device = XPtrTorchOptionalDevice;
using IntArrayRef = XPtrTorchOptionalIntArrayRef;
using Generator = XPtrTorchOptionalGenerator;

using int64_t = XPtrTorchoptional_int64_t;
using bool_t = XPtrTorchoptional_bool;

}  // namespace optional

namespace vector {
using Tensor = XPtrTorchTensorList;
using int64_t = XPtrTorchvector_int64_t;
using string = XPtrTorchvector_string;
using bool_t = XPtrTorchvector_bool;
using Scalar = XPtrTorchvector_Scalar;
using double_t = XPtrTorchvector_double;
using void_ptr = XPtrTorchvector_void;

namespace jit {
using IValue = XPtrTorchvector_IValue;
}
}  // namespace vector

namespace index {
using Tensor = XPtrTorchIndexTensor;
using TensorList = XPtrTorchIndexTensorList;
using IntArrayRef = XPtrTorchIndexIntArrayRef;
using int64_t = XPtrTorchindex_int64_t;
namespace optional {
using TensorList = XPtrTorchOptionalIndexTensorList;
using IntArrayRef = XPtrTorchOptionalIndexIntArrayRef;
}  // namespace optional
}  // namespace index

namespace jit {
using named_parameter_list = XPtrTorchjit_named_parameter_list;
using named_buffer_list = XPtrTorchjit_named_buffer_list;
using named_module_list = XPtrTorchjit_named_module_list;
using FunctionPtr = XPtrTorchFunctionPtr;
using Stack = XPtrTorchStack;
using IValue = XPtrTorchIValue;
using Tuple = XPtrTorchTuple;

namespace vector {
using IValue = XPtrTorchTuple;
}

namespace script {
using Module = XPtrTorchScriptModule;
using Method = XPtrTorchScriptMethod;
}  // namespace script

namespace impl {
using TraceableFunction = XPtrTorchTraceableFunction;
}

}  // namespace jit
}  // namespace torch

#endif  // TORCH_DYPES