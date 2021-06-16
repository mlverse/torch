#pragma once
#define LANTERN_HEADERS_ONLY
#include <string>
#include <memory>

#include "lantern/lantern.h"

#include <RcppCommon.h>

class XPtrTorch
{
private:
  std::shared_ptr<void> ptr;
public:
  XPtrTorch(void * x, std::function<void(void*)> deleter = [](void*){}) :
    XPtrTorch(std::shared_ptr<void>(x, deleter)) {};
  explicit XPtrTorch(std::shared_ptr<void> x) : ptr(x) {}
  void* get() const
  {
    return ptr.get();
  }
  std::shared_ptr<void> get_shared() const {
    return ptr;
  }
};

class XPtrTorchIndexTensor: public XPtrTorch {
public:
  XPtrTorchIndexTensor (): XPtrTorch{NULL} {}
  XPtrTorchIndexTensor (void* x) : XPtrTorch(x, lantern_Tensor_delete) {}
  explicit XPtrTorchIndexTensor (std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchIndexTensor (const XPtrTorchIndexTensor& x): XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchIndexTensor (SEXP x);
  operator SEXP () const;
};

std::function<void(void*)> tensor_deleter (void* x);

class XPtrTorchTensor : public XPtrTorch {
public:
  // TODO: we should make this explicit at some point, but not currently
  // possible because we rely on it in too many places.
  XPtrTorchTensor () : XPtrTorch{NULL} {}
  XPtrTorchTensor (void* x) : XPtrTorch(x, lantern_Tensor_delete) {}
  explicit XPtrTorchTensor (std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchTensor (const XPtrTorchTensor& x): XPtrTorch(x.get_shared()) {}
  XPtrTorchTensor (XPtrTorchIndexTensor x): XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchTensor (SEXP x);
  operator SEXP () const;
};

class XPtrTorchOptionalTensor : public XPtrTorch {
public:
  XPtrTorchOptionalTensor (void* x) : XPtrTorch(x, lantern_optional_tensor_delete) {}
  explicit XPtrTorchOptionalTensor (std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchOptionalTensor (const XPtrTorchOptionalTensor& x): XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchOptionalTensor (SEXP x);
  operator SEXP () const;
};

class XPtrTorchTensorList : public XPtrTorch {
public:
  XPtrTorchTensorList (void* x) : XPtrTorch(x, lantern_TensorList_delete) {}
  explicit XPtrTorchTensorList (std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchTensorList (const XPtrTorchTensorList& x) : XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchTensorList (SEXP x);
  operator SEXP () const;
};

class XPtrTorchOptionalTensorList : public XPtrTorch {
public:
  XPtrTorchOptionalTensorList (void* x) : XPtrTorch(x, lantern_TensorList_delete) {}
  explicit XPtrTorchOptionalTensorList (std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchOptionalTensorList (const XPtrTorchTensorList& x) : XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchOptionalTensorList (SEXP x);
  operator SEXP () const;
};

class XPtrTorchIndexTensorList: public XPtrTorch {
public:
  XPtrTorchIndexTensorList (void* x) : XPtrTorch(x, lantern_TensorList_delete) {}
  explicit XPtrTorchIndexTensorList (std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchIndexTensorList (const XPtrTorchIndexTensorList& x) : XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchIndexTensorList (SEXP x);
};

class XPtrTorchScalarType : public XPtrTorch {
public:
  XPtrTorchScalarType (void* x) : XPtrTorch (x, lantern_ScalarType_delete) {}
  explicit XPtrTorchScalarType (std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchScalarType (const XPtrTorchScalarType& x) : XPtrTorch(x.get_shared()) {}
  operator SEXP () const;
};

class XPtrTorchScalar : public XPtrTorch {
public:
  XPtrTorchScalar () : XPtrTorch{NULL} {}
  XPtrTorchScalar (void* x) : XPtrTorch(x, lantern_Scalar_delete) {}
  explicit XPtrTorchScalar (std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchScalar (const XPtrTorchScalar& x) : XPtrTorch(x.get_shared()) {}
  explicit XPtrTorchScalar (SEXP x);
  operator SEXP () const;
};

class XPtrTorchTensorOptions : public XPtrTorch {
public:
  XPtrTorchTensorOptions (void* x) : XPtrTorch(x, lantern_TensorOptions_delete) {};
  explicit XPtrTorchTensorOptions (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchTensorOptions (const XPtrTorchTensorOptions& x) : XPtrTorch(x.get_shared()) {};
  explicit XPtrTorchTensorOptions (SEXP x);
  operator SEXP () const;
};

class XPtrTorchDevice : public XPtrTorch {
public:
  XPtrTorchDevice (void* x) : XPtrTorch(x, lantern_Device_delete) {}
  explicit XPtrTorchDevice (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchDevice (const XPtrTorchDevice& x) : XPtrTorch(x.get_shared()) {};
  explicit XPtrTorchDevice (SEXP x);
  operator SEXP () const;
};

class XPtrTorchOptionalDevice : public XPtrTorch {
public:
  XPtrTorchOptionalDevice (void* x) : XPtrTorch(x, lantern_optional_device_delete) {}
  explicit XPtrTorchOptionalDevice (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchOptionalDevice (const XPtrTorchOptionalDevice& x) : XPtrTorch(x.get_shared()) {};
  explicit XPtrTorchOptionalDevice (SEXP x);
  operator SEXP () const;
};

class XPtrTorchDtype : public XPtrTorch {
public:
  XPtrTorchDtype () : XPtrTorch{NULL} {}
  XPtrTorchDtype (void* x) : XPtrTorch(x, lantern_Dtype_delete) {}
  explicit XPtrTorchDtype (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchDtype (const XPtrTorchDtype& x) : XPtrTorch(x.get_shared()) {};
  explicit XPtrTorchDtype (SEXP x);
  operator SEXP () const;
};

class XPtrTorchDimname : public XPtrTorch {
public:
  XPtrTorchDimname (void* x) : XPtrTorch(x, lantern_Dimname_delete) {}
  explicit XPtrTorchDimname (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchDimname (const XPtrTorchDimname& x) : XPtrTorch(x.get_shared()) {};
  explicit XPtrTorchDimname (SEXP x);
  explicit XPtrTorchDimname (const std::string& x) : 
    XPtrTorch(lantern_Dimname(x.c_str()), lantern_Dimname_delete) {};
  operator SEXP () const;
};

class XPtrTorchDimnameList : public XPtrTorch {
public:
  XPtrTorchDimnameList (void* x) : XPtrTorch(x, lantern_DimnameList_delete) {}
  explicit XPtrTorchDimnameList (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchDimnameList (const XPtrTorchDimnameList& x) : XPtrTorch(x.get_shared()) {};
  explicit XPtrTorchDimnameList (SEXP x);
  operator SEXP () const;
};

class XPtrTorchjit_named_parameter_list : public XPtrTorch {
public:
  XPtrTorchjit_named_parameter_list (void* x) : XPtrTorch(x, lantern_jit_named_parameter_list_delete) {}
  explicit XPtrTorchjit_named_parameter_list (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchjit_named_parameter_list (const XPtrTorchjit_named_parameter_list& x) : XPtrTorch(x.get_shared()) {};
  operator SEXP () const;
};

class XPtrTorchGenerator : public XPtrTorch {
public:
  XPtrTorchGenerator (void* x) : XPtrTorch(x, lantern_Generator_delete) {}
  explicit XPtrTorchGenerator (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchGenerator (const XPtrTorchGenerator& x) : XPtrTorch(x.get_shared()) {};
  explicit XPtrTorchGenerator (SEXP x);
  operator SEXP () const;
};

class XPtrTorchMemoryFormat : public XPtrTorch {
public:
  XPtrTorchMemoryFormat () : XPtrTorch{NULL} {};
  XPtrTorchMemoryFormat (void* x) : XPtrTorch(x, lantern_MemoryFormat_delete) {}
  explicit XPtrTorchMemoryFormat (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchMemoryFormat (const XPtrTorchMemoryFormat& x) : XPtrTorch(x.get_shared()) {};
  explicit XPtrTorchMemoryFormat (SEXP x);
  operator SEXP () const;
};

class XPtrTorchIntArrayRef : public XPtrTorch {
public:
  XPtrTorchIntArrayRef () : XPtrTorch{NULL} {};
  XPtrTorchIntArrayRef (void* x) : XPtrTorch(x, lantern_vector_int64_t_delete) {}
  explicit XPtrTorchIntArrayRef (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchIntArrayRef (const XPtrTorchIntArrayRef& x) : XPtrTorch(x.get_shared()) {};
  explicit XPtrTorchIntArrayRef (SEXP x);
  //operator SEXP () const;
};

class XPtrTorchIndexIntArrayRef : public XPtrTorch {
public:
  XPtrTorchIndexIntArrayRef () : XPtrTorch{NULL} {};
  XPtrTorchIndexIntArrayRef (void* x) : XPtrTorch(x, lantern_vector_int64_t_delete) {}
  explicit XPtrTorchIndexIntArrayRef (std::shared_ptr<void> x) : XPtrTorch(x) {};
  XPtrTorchIndexIntArrayRef (const XPtrTorchIndexIntArrayRef& x) : XPtrTorch(x.get_shared()) {};
  explicit XPtrTorchIndexIntArrayRef (SEXP x);
};

class XPtrTorchOptionalIntArrayRef {
public:
  std::shared_ptr<void> ptr;
  std::vector<int64_t> data;
  bool is_null;
  
  XPtrTorchOptionalIntArrayRef () {};
  explicit XPtrTorchOptionalIntArrayRef (SEXP x);
  
  XPtrTorchOptionalIntArrayRef (std::vector<int64_t> data_, bool is_null_) {
    data = data_;
    ptr = std::shared_ptr<void>(
      lantern_optional_vector_int64_t(data.data(), data.size(), is_null_),
      lantern_optional_vector_int64_t_delete
    );
    is_null = is_null_;
  }
  
  XPtrTorchOptionalIntArrayRef (const XPtrTorchOptionalIntArrayRef& x ) : 
    XPtrTorchOptionalIntArrayRef (x.data, x.is_null) {}
  
  void* get() const
  {
    return ptr.get();
  }
};  

class XPtrTorchOptionalIndexIntArrayRef : public XPtrTorchOptionalIntArrayRef {
public:
  explicit XPtrTorchOptionalIndexIntArrayRef (SEXP x);
};

class XPtrTorchint64_t2 {
public:
  std::shared_ptr<void> ptr;
  explicit XPtrTorchint64_t2 (SEXP x_);
  void* get () {
    return ptr.get();
  }
};

class XPtrTorchoptional_int64_t2 {
public:
  std::shared_ptr<void> ptr;
  explicit XPtrTorchoptional_int64_t2 (SEXP x_);
  void* get () {
    return ptr.get();
  }
};

class XPtrTorchindex_int64_t {
public:
  std::shared_ptr<void> ptr;
  explicit XPtrTorchindex_int64_t (SEXP x_);
  void* get () {
    return ptr.get();
  }
};

class XPtrTorchoptional_index_int64_t {
public:
  std::shared_ptr<void> ptr;
  explicit XPtrTorchoptional_index_int64_t (SEXP x_);
  void* get () {
    return ptr.get();
  }
};

class XPtrTorchvector_string : public XPtrTorch {
public:
  XPtrTorchvector_string (void * x) : XPtrTorch(x, lantern_vector_string_delete) {}
  operator SEXP () const;
};

#include <Rcpp.h>

class XPtrTorchQScheme : public XPtrTorch {
public:
  XPtrTorchQScheme (void* x) : XPtrTorch (x, lantern_QScheme_delete) {}
};

class XPtrTorchdouble : public XPtrTorch {
public:
  XPtrTorchdouble (void* x) : XPtrTorch(x, lantern_double_delete) {}
};

class XPtrTorchvariable_list : public XPtrTorch {
public:
  XPtrTorchvariable_list (void* x) : XPtrTorch(x, lantern_variable_list_delete) {}
};

class XPtrTorchint64_t : public XPtrTorch {
public:
  XPtrTorchint64_t (void* x) : XPtrTorch(x, lantern_int64_t_delete) {}
};

class XPtrTorchvector_int64_t : public XPtrTorch {
public:
  XPtrTorchvector_int64_t (void* x) : XPtrTorch(x, lantern_vector_int64_t_delete) {}
};

class XPtrTorchbool : public XPtrTorch {
public:
  XPtrTorchbool (void* x) : XPtrTorch(x, lantern_bool_delete) {}
};


class XPtrTorchLayout : public XPtrTorch {
public:
  XPtrTorchLayout (void* x) : XPtrTorch(x, lantern_Layout_delete) {}
};

class XPtrTorchTensorIndex : public XPtrTorch {
public:
  XPtrTorchTensorIndex (void* x) : XPtrTorch(x, lantern_TensorIndex_delete) {}
};

class XPtrTorchoptional_int64_t : public XPtrTorch {
public:
  XPtrTorchoptional_int64_t (void* x) : XPtrTorch(x, lantern_optional_int64_t_delete) {}
};

class XPtrTorchSlice : public XPtrTorch {
public:
  XPtrTorchSlice (void* x) : XPtrTorch(x, lantern_Slice_delete) {}
};

class XPtrTorchPackedSequence : public XPtrTorch {
public:
  XPtrTorchPackedSequence (void * x) : XPtrTorch(x, lantern_PackedSequence_delete) {}
};

class XPtrTorchStorage : public XPtrTorch {
public:
  XPtrTorchStorage (void * x) : XPtrTorch(x, lantern_Storage_delete) {}
};

class XPtrTorchIValue : public XPtrTorch {
public:
  XPtrTorchIValue (void * x) : XPtrTorch (x, lantern_IValue_delete) {}
};

class XPtrTorchstring : public XPtrTorch {
public:
  XPtrTorchstring (void * x) : XPtrTorch(x, lantern_string_delete) {}
};

class XPtrTorchStack : public XPtrTorch {
public:
  XPtrTorchStack (void * x) : XPtrTorch(x, lantern_Stack_delete) {}
};

class XPtrTorchCompilationUnit : public XPtrTorch {
public:
  XPtrTorchCompilationUnit (void * x) : XPtrTorch(x, lantern_CompilationUnit_delete) {}
};

class XPtrTorchJITModule : public XPtrTorch {
public:
  XPtrTorchJITModule (void * x) : XPtrTorch(x, lantern_JITModule_delete) {}
};

class XPtrTorchTraceableFunction : public XPtrTorch {
public:
  XPtrTorchTraceableFunction (void * x) : XPtrTorch(x, lantern_TraceableFunction_delete) {}
};

class XPtrTorchvector_bool : public XPtrTorch {
public:
  XPtrTorchvector_bool (void * x) : XPtrTorch(x, lantern_vector_bool_delete) {}
};

class XPtrTorchvector_void : public XPtrTorch {
public:
  XPtrTorchvector_void (void * x) : XPtrTorch(x, lantern_vector_void_delete) {}
};

template<class T>
class nullable {
public:
  T x;
  bool is_null = false;
  nullable (Rcpp::Nullable<T> x) {
    if (x.isNotNull()) {
      this->x = Rcpp::as<T>(x);
    } else {
      this->is_null = true;
    }
  };
  void* get () {
    if (this->is_null)
      return nullptr;
    else
      return &(this->x);
  }
  T get_value () {
    if (this->is_null)
      return NULL;
    else
      return this->x;
  }
};

template<class T>
class nullableVector {
public:
  T x = {0};
  bool is_null = false;
  nullableVector (Rcpp::Nullable<T> x) {
    if (x.isNotNull()) {
      this->x = Rcpp::as<T>(x);
    } else {
      this->is_null = true;
    }
  };
  void* get () {
    if (this->is_null)
      return nullptr;
    else
      return &(this->x);
  }
  T get_value () {
    if (this->is_null)
      return NULL;
    else
      return this->x;
  }
};
