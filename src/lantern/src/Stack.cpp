#define LANTERN_BUILD
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"

using namespace torch::jit::tracer;
using namespace torch::jit;

void* _lantern_Stack_new() {
  LANTERN_FUNCTION_START;
  return make_ptr<torch::jit::Stack>();
  LANTERN_FUNCTION_END;
}

int64_t _lantern_Stack_size(void* self) {
  LANTERN_FUNCTION_START;
  return reinterpret_cast<torch::jit::Stack*>(self)->size();
  LANTERN_FUNCTION_END;
}

void* _lantern_Stack_at(void* self, int64_t index) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<torch::jit::Stack*>(self);
  return make_raw::IValue(self_->at(index));
  LANTERN_FUNCTION_END
}

void _lantern_Stack_push_back_IValue(void* self, void* x) {
  LANTERN_FUNCTION_START;
  reinterpret_cast<torch::jit::Stack*>(self)->push_back(from_raw::IValue(x));
  LANTERN_FUNCTION_END_VOID;
}