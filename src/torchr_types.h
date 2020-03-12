#pragma once

class XPtrTorch
{
private:
  void* ptr;
public:
  XPtrTorch(void* value)
  {
    ptr = value;
  }
  void* get()
  {
    return ptr;
  }
};

#define LANTERN_HEADERS_ONLY
#include <string>
#include "lantern/lantern.h"
#include <Rcpp.h>

class XPtrTorchTensor : public XPtrTorch {
public:
  using XPtrTorch::XPtrTorch;
  ~XPtrTorchTensor () {
    Rcpp::Rcout << "deleting" << std::endl;
    lantern_Tensor_delete(get());
  }
};

class XPtrTorchScalarType: public XPtrTorch {
public:
  using XPtrTorch::XPtrTorch;
  ~XPtrTorchScalarType () {
    Rcpp::Rcout << "deleting" << std::endl;
    lantern_ScalarType_delete(get());
  }
};

class XPtrTorchScalar: public XPtrTorch {
public:
  using XPtrTorch::XPtrTorch;
  ~XPtrTorchScalar () {
    Rcpp::Rcout << "deleting" << std::endl;
    lantern_Scalar_delete(get());
  }
};

class XPtrTorchQScheme: public XPtrTorch {
public:
  using XPtrTorch::XPtrTorch;
  ~XPtrTorchQScheme () {
    Rcpp::Rcout << "deleting" << std::endl;
    lantern_QScheme_delete(get());
  }
};

class XPtrTorchdouble: public XPtrTorch {
public:
  using XPtrTorch::XPtrTorch;
  ~XPtrTorchdouble () {
    Rcpp::Rcout << "deleting" << std::endl;
    lantern_double_delete(get());
  }
};

class XPtrTorchTensorList: public XPtrTorch {
public:
  using XPtrTorch::XPtrTorch;
  ~XPtrTorchTensorList () {
    Rcpp::Rcout << "deleting" << std::endl;
    lantern_TensorList_delete(get());
  }
};

class XPtrTorchint64_t: public XPtrTorch {
public:
  using XPtrTorch::XPtrTorch;
  ~XPtrTorchint64_t () {
    Rcpp::Rcout << "deleting" << std::endl;
    lantern_int64_t_delete(get());
  }
};

class XPtrTorchbool : public XPtrTorch {
public:
  using XPtrTorch::XPtrTorch;
  ~XPtrTorchbool () {
    Rcpp::Rcout << "deleting" << std::endl;
    lantern_bool_delete(get());
  }
};