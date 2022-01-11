#include <iostream>

#define LANTERN_BUILD

#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"

#define LANTERN_DTYPE_FUN(name, type) \
  void *_lantern_Dtype_##name() { return make_raw::Dtype(torch::type); }

LANTERN_DTYPE_FUN(float16, kFloat16)
LANTERN_DTYPE_FUN(float32, kFloat32)
LANTERN_DTYPE_FUN(float64, kFloat64)
LANTERN_DTYPE_FUN(int8, kInt8)
LANTERN_DTYPE_FUN(int16, kInt16)
LANTERN_DTYPE_FUN(int32, kInt32)
LANTERN_DTYPE_FUN(int64, kInt64)
LANTERN_DTYPE_FUN(uint8, kUInt8)
LANTERN_DTYPE_FUN(bool, kBool)
LANTERN_DTYPE_FUN(quint8, kQUInt8)
LANTERN_DTYPE_FUN(qint8, kQInt8)
LANTERN_DTYPE_FUN(qint32, kQInt32)

const char *_lantern_Dtype_type(void *dtype) {
  LANTERN_FUNCTION_START
  std::string str = toString(from_raw::Dtype(dtype));
  char *cstr = new char[str.length() + 1];
  strcpy(cstr, str.c_str());
  return cstr;
  LANTERN_FUNCTION_END
}

void _lantern_set_default_dtype(void *dtype) {
  LANTERN_FUNCTION_START
  torch::Dtype dt = from_raw::Dtype(dtype);
  torch::set_default_dtype(c10::scalarTypeToTypeMeta(dt));
  LANTERN_FUNCTION_END_VOID
}

void *_lantern_get_default_dtype() {
  LANTERN_FUNCTION_START
  auto dt = torch::get_default_dtype();
  return make_raw::Dtype(c10::typeMetaToScalarType(dt));
  LANTERN_FUNCTION_END
}
