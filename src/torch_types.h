#pragma once

#include <Rcpp.h>

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
#include "lantern/lantern.h"