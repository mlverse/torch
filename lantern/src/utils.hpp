#pragma once
#include "lantern/types.h"

template <class T>
class LanternPtr {
 private:
  T *_object;

 public:
  LanternPtr(const T &object) { _object = new T(object); }

  LanternPtr() { _object = new T; }

  ~LanternPtr() {
    delete _object;
    _object = NULL;
  }

  T &get() { return *_object; }

  operator T &() const { return *_object; }
};
