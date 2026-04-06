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
LANTERN_DTYPE_FUN(bfloat16, kBFloat16)
LANTERN_DTYPE_FUN(int8, kInt8)
LANTERN_DTYPE_FUN(int16, kInt16)
LANTERN_DTYPE_FUN(int32, kInt32)
LANTERN_DTYPE_FUN(int64, kInt64)
LANTERN_DTYPE_FUN(uint8, kUInt8)
LANTERN_DTYPE_FUN(bool, kBool)
LANTERN_DTYPE_FUN(quint8, kQUInt8)
LANTERN_DTYPE_FUN(qint8, kQInt8)
LANTERN_DTYPE_FUN(qint32, kQInt32)
LANTERN_DTYPE_FUN(chalf, kComplexHalf)
LANTERN_DTYPE_FUN(cfloat, kComplexFloat)
LANTERN_DTYPE_FUN(cdouble, kComplexDouble)
LANTERN_DTYPE_FUN(byte, kByte)
LANTERN_DTYPE_FUN(float8_e4m3fn, kFloat8_e4m3fn)
LANTERN_DTYPE_FUN(float8_e5m2, kFloat8_e5m2)
    
void* _lantern_Dtype_from_string (void* dtype_str) {
  LANTERN_FUNCTION_START
  
  if (!dtype_str) {
    throw std::runtime_error("Error dtype can't be NULL");
  }
  
  auto str = from_raw::string(dtype_str);
  auto dtype = [&str] () {
    if (str == "float" || str == "float32") {
      return torch::kFloat32;
    } else if (str == "float64" || str == "double") {
      return torch::kFloat64;
    } else if (str == "float16" || str == "half") {
      return torch::kFloat16;
    } else if (str == "bfloat16") {
      return at::kBFloat16;
    } else if (str == "complex32" || str == "chalf") {
      return torch::kComplexHalf;
    } else if (str == "complex64" || str == "cfloat") {
      return torch::kComplexFloat;
    } else if (str == "complex128" || str == "cdouble") {
      return torch::kComplexDouble;
    } else if (str == "uint8") {
      return torch::kByte;
    } else if (str == "int8") {
      return torch::kInt8;
    } else if (str == "int16" || str == "short") {
      return torch::kInt16;
    } else if (str == "int32" || str == "int") {
      return torch::kInt32;
    } else if (str == "int64" || str == "long") {
      return torch::kInt64;
    } else if (str == "bool") {
      return torch::kBool;
    } else if (str == "quint8") {
      return torch::kQUInt8;
    } else if (str == "qint8") {
      return torch::kQInt8;
    } else if (str == "qint32") {
      return torch::kQInt32;
    } else if (str == "quint4x2") {
      return torch::kQUInt4x2;
    } else if (str == "float8_e4m3fn") {
      return torch::kFloat8_e4m3fn;
    } else if (str == "float8_e5m2") {
      return torch::kFloat8_e5m2;
    } else {
      throw std::runtime_error("Error unknown type " + str);
    }
  }();
  return make_raw::Dtype(dtype);
  LANTERN_FUNCTION_END
}

void* _lantern_Dtype_type(void *dtype) {
  LANTERN_FUNCTION_START
  std::string str = toString(from_raw::Dtype(dtype));
  return make_raw::string(str);
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
