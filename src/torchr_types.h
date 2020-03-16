#pragma once
#define LANTERN_HEADERS_ONLY
#include <string>
#include <memory>
#include "lantern/lantern.h"
#include <Rcpp.h>

class XPtrTorch
{
private:
  std::shared_ptr<void> ptr;
public:
  XPtrTorch (void * x) {
    this->set(std::shared_ptr<void>(x, [](void*){}));
  }
  XPtrTorch (std::shared_ptr<void> x) {
    this->set(x);
  }
  void* get()
  {
    return ptr.get();
  }
  void set (std::shared_ptr<void> x) {
    this->ptr = x;
  }
};

class XPtrTorchTensor : public XPtrTorch {
public:
  XPtrTorchTensor (void* x) : XPtrTorch{NULL} {
    this->set(std::shared_ptr<void>(x, lantern_Tensor_delete)); 
  }
};

class XPtrTorchScalarType : public XPtrTorch {
public:
  XPtrTorchScalarType (void* x) : XPtrTorch{NULL} {
    this->set(std::shared_ptr<void>(x, lantern_ScalarType_delete));
  }
};

class XPtrTorchScalar : public XPtrTorch {
public:
  XPtrTorchScalar () : XPtrTorch{NULL} {
    // do nothing
  }
  XPtrTorchScalar (void* x) : XPtrTorch{NULL} {
    this->set(std::shared_ptr<void>(x, lantern_Scalar_delete));
  }
};

class XPtrTorchQScheme : public XPtrTorch {
public:
  XPtrTorchQScheme (void* x) : XPtrTorch{NULL} {
    this->set(std::shared_ptr<void>(x, lantern_QScheme_delete));
  }
};

class XPtrTorchdouble : public XPtrTorch {
public:
  XPtrTorchdouble (void* x) : XPtrTorch{NULL} {
    this->set(std::shared_ptr<void>(x, lantern_double_delete));
  }
};

class XPtrTorchTensorList : public XPtrTorch {
public:
  XPtrTorchTensorList (void* x) : XPtrTorch{NULL} {
    this->set(std::shared_ptr<void>(x, lantern_TensorList_delete));
  }
};

class XPtrTorchint64_t : public XPtrTorch {
public:
  XPtrTorchint64_t (void* x) : XPtrTorch{NULL} {
    this->set(std::shared_ptr<void>(x, lantern_int64_t_delete));
  }
};

class XPtrTorchbool : public XPtrTorch {
public:
  XPtrTorchbool (void* x) : XPtrTorch{NULL} {
    this->set(std::shared_ptr<void>(x, lantern_bool_delete));
  }
};

class XPtrTorchTensorOptions : public XPtrTorch {
public:
  XPtrTorchTensorOptions (void* x) : XPtrTorch{NULL} {
    this->set(std::shared_ptr<void>(x, lantern_TensorOptions_delete));
  }
};

class XPtrTorchDevice : public XPtrTorch {
public:
  XPtrTorchDevice (void* x) : XPtrTorch{NULL} {
    this->set(std::shared_ptr<void>(x, lantern_Device_delete));
  }
};

class XPtrTorchLayout : public XPtrTorch {
public:
  XPtrTorchLayout (void* x) : XPtrTorch{NULL} {
    this->set(std::shared_ptr<void>(x, lantern_Layout_delete));
  }
};

class XPtrTorchDtype : public XPtrTorch {
public:
  XPtrTorchDtype (void* x) : XPtrTorch{NULL} {
    this->set(std::shared_ptr<void>(x, lantern_Dtype_delete));
  }
};

class XPtrTorchGenerator : public XPtrTorch {
public:
  XPtrTorchGenerator (void* x) : XPtrTorch{NULL} {
    this->set(std::shared_ptr<void>(x, lantern_Generator_delete));
  }
};

class XPtrTorchDimname : public XPtrTorch {
public:
  XPtrTorchDimname (void* x) : XPtrTorch{NULL} {
    this->set(std::shared_ptr<void>(x, lantern_Dimname_delete));
  }
};

class XPtrTorchDimnameList : public XPtrTorch {
public:
  XPtrTorchDimnameList (void* x) : XPtrTorch{NULL} {
    this->set(std::shared_ptr<void>(x, lantern_DimnameList_delete));
  }
};

class XPtrTorchMemoryFormat : public XPtrTorch {
public:
  XPtrTorchMemoryFormat (void* x) : XPtrTorch{NULL} {
    this->set(std::shared_ptr<void>(x, lantern_MemoryFormat_delete));
  }
};

