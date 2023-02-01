#include <iostream>

#define LANTERN_BUILD

#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"

using namespace torch::indexing;

void *_lantern_TensorIndex_new() {
  LANTERN_FUNCTION_START
  return make_ptr<std::vector<at::indexing::TensorIndex>>();
  LANTERN_FUNCTION_END
}

bool _lantern_TensorIndex_is_empty(void *self) {
  LANTERN_FUNCTION_START
  return reinterpret_cast<std::vector<at::indexing::TensorIndex> *>(self)
             ->size() == 0;
  LANTERN_FUNCTION_END
}

void _lantern_TensorIndex_append_tensor(void *self, void *x) {
  LANTERN_FUNCTION_START
  torch::Tensor ten = from_raw::Tensor(x);
  reinterpret_cast<std::vector<at::indexing::TensorIndex> *>(self)->push_back(
      ten);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_TensorIndex_append_ellipsis(void *self) {
  LANTERN_FUNCTION_START
  reinterpret_cast<std::vector<at::indexing::TensorIndex> *>(self)->push_back(
      Ellipsis);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_TensorIndex_append_slice(void *self, void *x) {
  LANTERN_FUNCTION_START
  torch::indexing::Slice slice = *reinterpret_cast<torch::indexing::Slice *>(x);
  reinterpret_cast<std::vector<at::indexing::TensorIndex> *>(self)->push_back(
      slice);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_TensorIndex_append_none(void *self) {
  LANTERN_FUNCTION_START
  reinterpret_cast<std::vector<at::indexing::TensorIndex> *>(self)->push_back(
      None);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_TensorIndex_append_bool(void *self, bool x) {
  LANTERN_FUNCTION_START
  reinterpret_cast<std::vector<at::indexing::TensorIndex> *>(self)->push_back(
      x);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_TensorIndex_append_int64(void *self, int64_t x) {
  LANTERN_FUNCTION_START
  reinterpret_cast<std::vector<at::indexing::TensorIndex> *>(self)->push_back(
      x);
  LANTERN_FUNCTION_END_VOID
}

void *_lantern_Slice(void *start, void *end, void *step) {
  LANTERN_FUNCTION_START
  auto start_ = from_raw::optional::int64_t(start);
  if (start_ == c10::nullopt) {
    start_ = None;
  }

  auto end_ = from_raw::optional::int64_t(end);
  if (end_ == c10::nullopt) {
    end_ = None;
  }

  auto step_ = from_raw::optional::int64_t(step);
  if (step_ == c10::nullopt) {
    step_ = None;
  }

  auto out = torch::indexing::Slice(start_, end_, step_);
  return make_ptr<torch::indexing::Slice>(out);
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_index(void *self, void *index) {
  LANTERN_FUNCTION_START
  torch::Tensor ten = from_raw::Tensor(self);
  torch::Tensor out = ten.index(
      *reinterpret_cast<std::vector<at::indexing::TensorIndex> *>(index));
  return make_raw::Tensor(out);
  LANTERN_FUNCTION_END
}

void _lantern_Tensor_index_put_tensor_(void *self, void *index, void *rhs) {
  LANTERN_FUNCTION_START
  torch::Tensor ten = from_raw::Tensor(self);
  auto i = *reinterpret_cast<std::vector<at::indexing::TensorIndex> *>(index);
  torch::Tensor r = from_raw::Tensor(rhs);
  ten.index_put_(i, r);
  LANTERN_FUNCTION_END_VOID
}

void _lantern_Tensor_index_put_scalar_(void *self, void *index, void *rhs) {
  LANTERN_FUNCTION_START
  torch::Tensor ten = from_raw::Tensor(self);
  auto i = *reinterpret_cast<std::vector<at::indexing::TensorIndex> *>(index);
  auto r = from_raw::Scalar(rhs);
  ten.index_put_(i, r);
  LANTERN_FUNCTION_END_VOID
}
