#define LANTERN_BUILD
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>

#include <iterator>

#include "lantern/lantern.h"
#include "utils.hpp"

using namespace torch::jit::tracer;
using namespace torch::jit;

template <class T>
int jit_type_size(void* self) {
  auto self_ = reinterpret_cast<T*>(self);
  return self_->size();
}

int _lantern_jit_named_parameter_list_size(void* self) {
  LANTERN_FUNCTION_START
  return jit_type_size<torch::jit::named_parameter_list>(self);
  LANTERN_FUNCTION_END
}

int _lantern_jit_named_module_list_size(void* self) {
  LANTERN_FUNCTION_START
  return jit_type_size<torch::jit::named_module_list>(self);
  LANTERN_FUNCTION_END
}

int _lantern_jit_named_buffer_list_size(void* self) {
  LANTERN_FUNCTION_START
  return jit_type_size<torch::jit::named_buffer_list>(self);
  LANTERN_FUNCTION_END
}

template <class T>
void* jit_named_list_tensors(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<T*>(self);
  std::vector<torch::Tensor> outputs;
  for (auto el : *self_) {
    outputs.push_back(el.value);
  }
  return make_raw::TensorList(outputs);
  LANTERN_FUNCTION_END
}

void* _lantern_jit_named_parameter_list_tensors(void* self) {
  LANTERN_FUNCTION_START
  return jit_named_list_tensors<torch::jit::named_parameter_list>(self);
  LANTERN_FUNCTION_END
}

void* _lantern_jit_named_buffer_list_tensors(void* self) {
  LANTERN_FUNCTION_START
  return jit_named_list_tensors<torch::jit::named_buffer_list>(self);
  LANTERN_FUNCTION_END
}

void* _lantern_jit_named_module_list_module_at(void* self, int64_t index) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<torch::jit::named_module_list*>(self);

  int i = 0;
  torch::jit::script::Module out;
  for (auto el : *self_) {
    if (i == index) {
      out = el.value;
      break;
    }
    i++;
  }

  return (void*)new torch::jit::script::Module(out);
  LANTERN_FUNCTION_END
}

template <class T>
void* jit_type_names(void* self) {
  auto self_ = reinterpret_cast<T*>(self);
  std::vector<std::string> outputs;
  for (auto el : *self_) {
    outputs.push_back(el.name);
  }
  return make_raw::vector::string(outputs);
}

void* _lantern_jit_named_parameter_list_names(void* self) {
  LANTERN_FUNCTION_START
  return jit_type_names<torch::jit::named_parameter_list>(self);
  LANTERN_FUNCTION_END
}

void* _lantern_jit_named_module_list_names(void* self) {
  LANTERN_FUNCTION_START
  return jit_type_names<torch::jit::named_module_list>(self);
  LANTERN_FUNCTION_END
}

void* _lantern_jit_named_buffer_list_names(void* self) {
  LANTERN_FUNCTION_START
  return jit_type_names<torch::jit::named_buffer_list>(self);
  LANTERN_FUNCTION_END
}

void* _lantern_jit_Tuple_new() {
  LANTERN_FUNCTION_START
  return (void*)new std::vector<torch::IValue>();
  LANTERN_FUNCTION_END
}

void _lantern_jit_Tuple_push_back(void* self, void* element) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<std::vector<torch::IValue>*>(self);
  self_->push_back(*reinterpret_cast<torch::IValue*>(element));
  LANTERN_FUNCTION_END_VOID
}

int64_t _lantern_jit_Tuple_size(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<std::vector<torch::IValue>*>(self);
  return self_->size();
  LANTERN_FUNCTION_END
}

void* _lantern_jit_Tuple_at(void* self, int64_t index) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<std::vector<torch::IValue>*>(self);
  return (void*)new torch::IValue(self_->at(index));
  LANTERN_FUNCTION_END
}

void* _lantern_jit_NamedTuple_new() {
  LANTERN_FUNCTION_START
  return (void*)new NamedTupleHelper();
  LANTERN_FUNCTION_END
}

void _lantern_jit_NamedTuple_push_back(void* self, void* name, void* element) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<NamedTupleHelper*>(self);
  self_->elements.push_back(*reinterpret_cast<torch::IValue*>(element));
  self_->names.push_back(*reinterpret_cast<std::string*>(name));
  LANTERN_FUNCTION_END_VOID
}

void* _lantern_jit_NamedTupleHelper_keys(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<NamedTupleHelper*>(self);
  return (void*)new std::vector<std::string>(self_->names);
  LANTERN_FUNCTION_END
}

void* _lantern_jit_NamedTupleHelper_elements(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<NamedTupleHelper*>(self);
  return (void*)new std::vector<torch::IValue>(self_->elements);
  LANTERN_FUNCTION_END
}

void* _lantern_jit_TensorDict_new() {
  LANTERN_FUNCTION_START
  return make_raw::TensorDict(alias::TensorDict());
  LANTERN_FUNCTION_END
}

void _lantern_jit_TensorDict_push_back(void* self, void* key, void* value) {
  LANTERN_FUNCTION_START
  auto self_ = from_raw::TensorDict(self);
  self_.insert(*reinterpret_cast<std::string*>(key), from_raw::Tensor(value));
  LANTERN_FUNCTION_END_VOID
}

void* _lantern_jit_GenericDict_keys(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = *reinterpret_cast<c10::impl::GenericDict*>(self);
  std::vector<torch::IValue> out;
  for (auto& element : self_) {
    out.push_back(element.key());
  }
  return (void*)new std::vector<torch::IValue>(out);
  LANTERN_FUNCTION_END
}

void* _lantern_jit_GenericDict_at(void* self, void* key) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<c10::impl::GenericDict*>(self);
  auto key_ = *reinterpret_cast<torch::IValue*>(key);
  return (void*)new torch::IValue(self_->at(key_));
  LANTERN_FUNCTION_END
}

int64_t _lantern_jit_GenericList_size(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<c10::impl::GenericList*>(self);
  return self_->size();
  LANTERN_FUNCTION_END
}

void* _lantern_jit_GenericList_at(void* self, int64_t index) {
  LANTERN_FUNCTION_START
  auto self_ = *reinterpret_cast<c10::impl::GenericList*>(self);
  return (void*)new torch::IValue(self_[index]);
  LANTERN_FUNCTION_END
}

int64_t _lantern_jit_vector_IValue_size(void* self) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<std::vector<torch::IValue>*>(self);
  return self_->size();
  LANTERN_FUNCTION_END
}

void* _lantern_jit_vector_IValue_at(void* self, int64_t index) {
  LANTERN_FUNCTION_START
  auto self_ = reinterpret_cast<std::vector<torch::IValue>*>(self);
  return (void*)new torch::IValue(self_->at(index));
  LANTERN_FUNCTION_END
}