#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void* lantern_Dtype_float32 () {
  return (void *) new LanternObject<torch::Dtype>(torch::kFloat32);
}

void* lantern_Dtype_float64 () {
  return (void *) new LanternObject<torch::Dtype>(torch::kFloat64);
}

void* lantern_Dtype_float16 () {
  return (void *) new LanternObject<torch::Dtype>(torch::kFloat16);
}

void* lantern_Dtype_uint8 () {
  return (void *) new LanternObject<torch::Dtype>(torch::kUInt8);
}

void* lantern_Dtype_int8 () {
  return (void *) new LanternObject<torch::Dtype>(torch::kInt8);
}

void* lantern_Dtype_int16 () {
  return (void *) new LanternObject<torch::Dtype>(torch::kInt16);
}

void* lantern_Dtype_int32 () {
  return (void *) new LanternObject<torch::Dtype>(torch::kInt32);
}

void* lantern_Dtype_int64 () {
  return (void *) new LanternObject<torch::Dtype>(torch::kInt64);
}

void* lantern_Dtype_bool () {
  return (void *) new LanternObject<torch::Dtype>(torch::kBool);
}

const char * lantern_Dtype_type (void * dtype) {
  return toString(reinterpret_cast<LanternObject<torch::Dtype>*>(dtype)->get());
}
