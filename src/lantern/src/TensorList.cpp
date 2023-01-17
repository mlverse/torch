#include <iostream>

#define LANTERN_BUILD

#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"

void* _lantern_TensorList() {
  LANTERN_FUNCTION_START
  return make_raw::TensorList({});
  LANTERN_FUNCTION_END
}

void* _lantern_OptionalTensorList() {
  LANTERN_FUNCTION_START
  c10::List<c10::optional<torch::Tensor>> list;
  return make_raw::optional::TensorList(list);
  LANTERN_FUNCTION_END
}

void _lantern_TensorList_push_back(void* self, void* x) {
  LANTERN_FUNCTION_START
  torch::Tensor ten = from_raw::Tensor(x);
  // We need to use the raw object here because this operation is not allowed
  // from a torch::TensorList directly. This should be the **only** place we
  // ever modify the buffer in-place.
  reinterpret_cast<self_contained::TensorList*>(self)->push_back(ten);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_OptionalTensorList_push_back(void* self, void* x, bool is_null) {
  // TODO make this take an optional tensor directly.
  LANTERN_FUNCTION_START
  c10::optional<torch::Tensor> tensor;
  if (is_null) {
    tensor = c10::nullopt;
  } else {
    tensor = from_raw::Tensor(x);
  }

  from_raw::optional::TensorList(self).push_back(tensor);
  LANTERN_FUNCTION_END_VOID
}

int64_t _lantern_OptionalTensorList_size(void* self) {
  LANTERN_FUNCTION_START
  return from_raw::optional::TensorList(self).size();
  LANTERN_FUNCTION_END
}

void* _lantern_OptionalTensorList_at(void* self, int64_t i) {
  LANTERN_FUNCTION_START
  auto t = from_raw::optional::TensorList(self).get(i);
  return make_raw::Tensor(t.value());
  LANTERN_FUNCTION_END
}

bool _lantern_OptionalTensorList_at_is_null(void* self, int64_t i) {
  LANTERN_FUNCTION_START
  return !from_raw::optional::TensorList(self).get(i).has_value();
  LANTERN_FUNCTION_END
}

void* _lantern_TensorList_at(void* self, int64_t i) {
  LANTERN_FUNCTION_START
  torch::Tensor out = from_raw::TensorList(self).at(i);
  return make_raw::Tensor(out);
  LANTERN_FUNCTION_END
}

int64_t _lantern_TensorList_size(void* self) {
  LANTERN_FUNCTION_START
  return from_raw::TensorList(self).size();
  LANTERN_FUNCTION_END_RET(0)
}

void* _lantern_Stream() {
  c10::Stream x = c10::Stream(c10::Stream::Default(),
                              torch::Device(torch::DeviceType::CPU));
  return make_raw::Stream(x);
}