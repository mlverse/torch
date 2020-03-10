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

class torchTensor : public XPtrTorch {
public:
  using XPtrTorch::XPtrTorch;
  ~torchTensor () {
    Rcpp::Rcout << "deleting" << std::endl;
    lantern_Tensor_delete(get());
  }
};