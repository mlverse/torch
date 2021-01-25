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

class XPtrTorchTensor : public XPtrTorch {
public:
  // TODO: we should make this explicit at some point, but not currently
  // possible because we rely on it in too many places.
  XPtrTorchTensor (void* x) : XPtrTorch(x, lantern_Tensor_delete) {}
  explicit XPtrTorchTensor (std::shared_ptr<void> x) : XPtrTorch(x) {}
  XPtrTorchTensor (const XPtrTorchTensor& x): XPtrTorch(x.get_shared()) {}
  operator SEXP () const;
};

#include <Rcpp.h>

class XPtrTorchScalarType : public XPtrTorch {
public:
  XPtrTorchScalarType (void* x) : XPtrTorch (x, lantern_ScalarType_delete) {}
};

class XPtrTorchScalar : public XPtrTorch {
public:
  XPtrTorchScalar () : XPtrTorch{NULL} {}
  XPtrTorchScalar (void* x) : XPtrTorch(x, lantern_Scalar_delete) {}
};

class XPtrTorchQScheme : public XPtrTorch {
public:
  XPtrTorchQScheme (void* x) : XPtrTorch (x, lantern_QScheme_delete) {}
};

class XPtrTorchdouble : public XPtrTorch {
public:
  XPtrTorchdouble (void* x) : XPtrTorch(x, lantern_double_delete) {}
};

class XPtrTorchTensorList : public XPtrTorch {
public:
  XPtrTorchTensorList (void* x) : XPtrTorch(x, lantern_TensorList_delete) {}
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

class XPtrTorchTensorOptions : public XPtrTorch {
public:
  XPtrTorchTensorOptions (void* x) : XPtrTorch(x, lantern_TensorOptions_delete) {}
};

class XPtrTorchDevice : public XPtrTorch {
public:
  XPtrTorchDevice (void* x) : XPtrTorch(x, lantern_Device_delete) {}
};

class XPtrTorchLayout : public XPtrTorch {
public:
  XPtrTorchLayout (void* x) : XPtrTorch(x, lantern_Layout_delete) {}
};

class XPtrTorchDtype : public XPtrTorch {
public:
  XPtrTorchDtype (void* x) : XPtrTorch(x, lantern_Dtype_delete) {}
};

class XPtrTorchGenerator : public XPtrTorch {
public:
  XPtrTorchGenerator (void* x) : XPtrTorch(x, lantern_Generator_delete) {}
};

class XPtrTorchDimname : public XPtrTorch {
public:
  XPtrTorchDimname (void* x) : XPtrTorch(x, lantern_Dimname_delete) {}
};

class XPtrTorchDimnameList : public XPtrTorch {
public:
  XPtrTorchDimnameList (void* x) : XPtrTorch(x, lantern_DimnameList_delete) {}
};

class XPtrTorchMemoryFormat : public XPtrTorch {
public:
  XPtrTorchMemoryFormat (void* x) : XPtrTorch(x, lantern_MemoryFormat_delete) {}
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

class XPtrTorchvector_string : public XPtrTorch {
public:
  XPtrTorchvector_string (void * x) : XPtrTorch(x, lantern_vector_string_delete) {}
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
