#include <iostream>

#define LANTERN_BUILD

#include <torch/torch.h>

#include "lantern/lantern.h"
#define LANTERN_TYPES_IMPL
#include "utils.hpp"

void *_lantern_vector_int64_t(int64_t *x, size_t x_size) {
  LANTERN_FUNCTION_START
  auto out = std::vector<int64_t>(x, x + x_size);
  return make_raw::vector::int64_t(out);
  LANTERN_FUNCTION_END
}

int64_t _lantern_vector_int64_t_size(void *self) {
  LANTERN_FUNCTION_START
  return from_raw::vector::int64_t(self).size();
  LANTERN_FUNCTION_END
}

int64_t _lantern_vector_int64_t_at(void *self, int64_t index) {
  return from_raw::vector::int64_t(self).at(index);
}

void *_lantern_vector_double(double *x, size_t x_size) {
  LANTERN_FUNCTION_START
  auto out = std::vector<double>(x, x + x_size);
  return make_raw::vector::double_t(out);
  LANTERN_FUNCTION_END
}

void *_lantern_vector_double_new() { return (void *)new std::vector<double>(); }

void *_lantern_vector_int64_t_new() { return make_raw::vector::int64_t({}); }

void _lantern_vector_double_push_back(void *self, double x) {
  reinterpret_cast<std::vector<double> *>(self)->push_back(x);
}

void _lantern_vector_int64_t_push_back(void *self, int64_t x) {
  // we should push back using the raw rype exclusevely here. That's to
  // correctly update both the buffer and the array ref.
  reinterpret_cast<self_contained::vector::int64_t *>(self)->push_back(x);
}

double _lantern_vector_double_size(void *self) {
  LANTERN_FUNCTION_START
  return reinterpret_cast<std::vector<double> *>(self)->size();
  LANTERN_FUNCTION_END
}

double _lantern_vector_double_at(void *self, int64_t index) {
  return reinterpret_cast<std::vector<double> *>(self)->at(index);
}

void *_lantern_optional_vector_double(double *x, size_t x_size, bool is_null) {
  LANTERN_FUNCTION_START
  c10::optional<torch::ArrayRef<double>> out;

  if (is_null)
    out = c10::nullopt;
  else
    out = torch::ArrayRef<double>(x, x + x_size);
  ;

  return make_raw::optional::DoubleArrayRef(out);
  LANTERN_FUNCTION_END
}

void *_lantern_optional_vector_int64_t(int64_t *x, size_t x_size,
                                       bool is_null) {
  LANTERN_FUNCTION_START
  c10::optional<torch::ArrayRef<int64_t>> out;

  if (is_null)
    out = c10::nullopt;
  else
    out = torch::ArrayRef<int64_t>(x, x + x_size);

  return make_raw::optional::IntArrayRef(out);
  LANTERN_FUNCTION_END
}

#define LANTERN_OPTIONAL(name, type)                              \
  void *_lantern_optional_##name(void *obj) {                     \
    if (!obj) {                                                   \
      return make_raw::optional::type(c10::nullopt);              \
    }                                                             \
    return make_raw::optional::type(from_raw::type(obj));         \
  }                                                               \
  bool _lantern_optional_##name##_has_value(void *obj) {          \
    return from_raw::optional::type(obj).has_value();             \
  }                                                               \
  void *_lantern_optional_##name##_value(void *obj) {             \
    return make_raw::type(from_raw::optional::type(obj).value()); \
  }

LANTERN_OPTIONAL(dimname_list, DimnameList)
LANTERN_OPTIONAL(generator, Generator)
LANTERN_OPTIONAL(tensor, Tensor)
LANTERN_OPTIONAL(double, double_t)
LANTERN_OPTIONAL(int64_t, int64_t)
LANTERN_OPTIONAL(bool, bool_t)
LANTERN_OPTIONAL(scalar_type, ScalarType)
LANTERN_OPTIONAL(string, string)
LANTERN_OPTIONAL(string_view, string_view)
LANTERN_OPTIONAL(memory_format, MemoryFormat)
LANTERN_OPTIONAL(scalar, Scalar)
LANTERN_OPTIONAL(device, Device)

void *_lantern_int64_t(int64_t x) {
  LANTERN_FUNCTION_START
  return make_raw::int64_t(x);
  LANTERN_FUNCTION_END
}

void *_lantern_double(double x) {
  LANTERN_FUNCTION_START
  return make_raw::double_t(x);
  LANTERN_FUNCTION_END
}

void *_lantern_bool(bool x) {
  LANTERN_FUNCTION_START
  return make_raw::bool_t(x);
  LANTERN_FUNCTION_END
}

bool _lantern_bool_get(void *x) {
  LANTERN_FUNCTION_START
  return from_raw::bool_t(x);
  LANTERN_FUNCTION_END
}

int64_t _lantern_int64_t_get(void *x) {
  LANTERN_FUNCTION_START
  return from_raw::int64_t(x);
  LANTERN_FUNCTION_END
}

double _lantern_double_get(void *x) {
  LANTERN_FUNCTION_START
  return from_raw::double_t(x);
  LANTERN_FUNCTION_END
}

void *_lantern_vector_get(void *x, int i) {
  LANTERN_FUNCTION_START
  auto v = from_raw::tuple(x);
  return v.at(i);
  LANTERN_FUNCTION_END
}

void *_lantern_Tensor_undefined() {
  LANTERN_FUNCTION_START
  torch::Tensor x = {};
  return make_raw::Tensor(x);
  LANTERN_FUNCTION_END
}

void *_lantern_vector_string_new() {
  LANTERN_FUNCTION_START
  return make_raw::vector::string();
  LANTERN_FUNCTION_END
}

void _lantern_vector_string_push_back(void *self, const char *x) {
  LANTERN_FUNCTION_START
  from_raw::vector::string(self).push_back(std::string(x));
  LANTERN_FUNCTION_END_VOID
}

int64_t _lantern_vector_string_size(void *self) {
  LANTERN_FUNCTION_START
  return from_raw::vector::string(self).size();
  LANTERN_FUNCTION_END_RET(0)
}

const char *_lantern_vector_string_at(void *self, int64_t i) {
  LANTERN_FUNCTION_START
  auto str = from_raw::vector::string(self).at(i);
  char *cstr = new char[str.length() + 1];
  strcpy(cstr, str.c_str());
  return cstr;
  LANTERN_FUNCTION_END
}

void *_lantern_vector_bool_new() {
  LANTERN_FUNCTION_START
  return make_raw::vector::bool_t();
  LANTERN_FUNCTION_END
}

void _lantern_vector_bool_push_back(void *self, bool x) {
  LANTERN_FUNCTION_START
  from_raw::vector::bool_t(self).push_back(x);
  LANTERN_FUNCTION_END_VOID
}

int64_t _lantern_vector_bool_size(void *self) {
  LANTERN_FUNCTION_START
  return from_raw::vector::bool_t(self).size();
  LANTERN_FUNCTION_END_RET(0)
}

bool _lantern_vector_bool_at(void *self, int64_t i) {
  LANTERN_FUNCTION_START
  return from_raw::vector::bool_t(self).at(i);
  LANTERN_FUNCTION_END_RET(false)
}

void *_lantern_string_new(const char *value) {
  LANTERN_FUNCTION_START
  return make_raw::string(std::string(value));
  LANTERN_FUNCTION_END
}

void *_lantern_string_view_new(const char *value) {
  LANTERN_FUNCTION_START
  return make_raw::string_view(std::string(value));
  LANTERN_FUNCTION_END
}

const char *_lantern_string_get(void *self) {
  auto str = from_raw::string(self);
  char *cstr = new char[str.length() + 1];
  strcpy(cstr, str.c_str());
  return cstr;
}

void _lantern_print_stuff(void *x) {
  LANTERN_FUNCTION_START
  auto v = reinterpret_cast<LanternPtr<c10::optional<torch::Device>> *>(x);
  std::cout << v->get().value().type() << std::endl;
  LANTERN_FUNCTION_END_VOID
}

void lantern_host_handler() {}
bool lantern_loaded = false;
void check_lantern_loaded() {}

void *_lantern_nn_functional_pad_circular(void *input, void *padding) {
  LANTERN_FUNCTION_START
  auto input_ = from_raw::Tensor(input);
  auto padding_ = from_raw::IntArrayRef(padding);
  auto out = torch::nn::functional::_pad_circular(input_, padding_);
  return make_raw::Tensor(out);
  LANTERN_FUNCTION_END
}
